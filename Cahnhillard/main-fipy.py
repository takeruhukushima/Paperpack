import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from matplotlib import animation
from matplotlib.animation import PillowWriter

# Try to import FiPy and HOOMD, and provide messages if not found
try:
    from fipy import CellVariable, Grid2D, DiffusionTerm, TransientTerm, ImplicitSourceTerm
    # from fipy.solvers.pysparse import LinearLUSolver # Example of specific solver
    FIPY_AVAILABLE = True
except ImportError:
    FIPY_AVAILABLE = False
    print("FiPy library not found. Part 1 (Cahn-Hilliard simulation) will be skipped.")

try:
    import hoomd
    import hoomd.md
    HOOMD_AVAILABLE = True
except ImportError:
    HOOMD_AVAILABLE = False
    print("HOOMD-blue library not found. Part 3 (Molecular Dynamics simulation) will be skipped.")

# -----------------------------------------------------------------------------
# ■ グローバルパラメータ設定
# -----------------------------------------------------------------------------

# Cahn-Hilliard パラメータ
W_CH = 1.0
KAPPA = 0.5
MOBILITY = 1.0
CH_NX, CH_NY = 128, 128
CH_DX = 1.0
CH_L = CH_NX * CH_DX

CH_PHI0_MEAN = 0.5
CH_PHI0_FLUCTUATION = 0.05
CH_STEPS = 10000
CH_DT = 0.01           # 時間ステップをさらに小さくして安定性を試す
CH_ANIMATION_SAVE_INTERVAL = 50
CH_ANIMATION_FPS = 20

# Flory-Huggins パラメータ
CHI = 2.2
N_PARAM = 1.0

# HOOMD-blue パラメータ
HOOMD_NUM_PARTICLES_A = 2000; HOOMD_NUM_PARTICLES_B = 1000; HOOMD_BOX_SIZE = 40.0
HOOMD_SIM_STEPS = 10000; HOOMD_DT = 0.005
LJ_EPSILON_AA = 1.0; LJ_SIGMA_AA = 1.0; LJ_RCUT_AA = 2.5
LJ_EPSILON_BB = 1.0; LJ_SIGMA_BB = 1.0; LJ_RCUT_BB = 2.5
LJ_EPSILON_AB = 0.5; LJ_SIGMA_AB = 1.0; LJ_RCUT_AB = 2.5

# -----------------------------------------------------------------------------
# ■ 1. FiPyによるCahn–Hilliard方程式（スピノーダル分解）
# -----------------------------------------------------------------------------
if FIPY_AVAILABLE:
    def dfdphi_double_well(phi_var, W):
        if isinstance(phi_var, CellVariable):
            phi_val = phi_var.value
        else:
            phi_val = phi_var
        phi_c = np.clip(phi_val, 1e-4, 1.0 - 1e-4)
        return 2.0 * W * phi_c * (1.0 - phi_c) * (1.0 - 2.0 * phi_c)

    def run_cahn_hilliard_simulation_fipy(nx, ny, dx, phi0_mean, phi0_fluctuation,
                                     W_param, kappa_param, M_param, steps, dt_ch, animation_save_interval):
        print("Running Cahn-Hilliard simulation with FiPy (Double-Well Potential)...")
        mesh = Grid2D(dx=dx, nx=nx, ny=ny)

        rng_fipy = np.random.default_rng(12345)
        phi0_array = phi0_mean + phi0_fluctuation * (rng_fipy.standard_normal((nx,ny)) - 0.5) * 2
        phi0_array = np.clip(phi0_array, 1e-3, 1.0 - 1e-3)

        phi = CellVariable(mesh=mesh, value=phi0_array.ravel(), hasOld=True, name="phi")
        mu = CellVariable(mesh=mesh, name="chemical_potential", hasOld=True)
        
        # df/dphiを格納するCellVariableを定義
        dfdphi_source = CellVariable(mesh=mesh, name="dfdphi_source_term")

        eq_phi = TransientTerm(var=phi) == DiffusionTerm(coeff=M_param, var=mu)
        # 右辺で dfdphi_source (CellVariable) を使用
        eq_mu = (ImplicitSourceTerm(coeff=1.0, var=mu) + DiffusionTerm(coeff=kappa_param, var=phi) ==
                 dfdphi_source)
        eq = eq_phi & eq_mu

        phi_evolution_data = []
        phi_evolution_data.append(phi.value.copy().reshape((nx, ny)))

        print(f"Initial phi min: {phi.value.min():.4f}, max: {phi.value.max():.4f}, mean: {phi.value.mean():.4f}")

        for step in range(steps):
            phi.updateOld()
            mu.updateOld()

            # 各ステップでdf/dphiを計算し、CellVariableを更新
            current_dfdphi_values = dfdphi_double_well(phi, W_param)
            dfdphi_source.setValue(current_dfdphi_values)

            try:
                res = eq.solve(dt=dt_ch)
            except Exception as e:
                print(f"Error during FiPy solve at step {step+1}: {e}")
                print("Solver might be unstable. Try reducing dt or adjusting other parameters.")
                return None, phi_evolution_data

            if np.any(phi.value < -0.1) or np.any(phi.value > 1.1):
                print(f"Warning: phi out of bounds at step {step+1}. Min: {phi.value.min():.4f}, Max: {phi.value.max():.4f}")
                # 必要に応じて値をクリップ: phi.setValue(np.clip(phi.value, 0.0, 1.0))


            if (step + 1) % animation_save_interval == 0 or step == steps - 1:
                phi_evolution_data.append(phi.value.copy().reshape((nx, ny)))

            if (step + 1) % (steps // 20 if steps >=20 else 1) == 0 or step == steps - 1:
                print(f"C-H Step: {step+1}/{steps}, Min/Max phi: {phi.value.min():.4f}/{phi.value.max():.4f}, Mean phi: {phi.value.mean():.4f}")

        print("Cahn-Hilliard simulation finished.")
        print(f"Final phi min: {phi.value.min():.4f}, max: {phi.value.max():.4f}, mean: {phi.value.mean():.4f}")
        return phi, phi_evolution_data

    def visualize_and_save_cahn_hilliard_result(phi_variable, nx, ny, output_dir, title="Cahn-Hilliard Result"):
        if phi_variable is None:
            print("No Cahn-Hilliard data to visualize/save (phi_variable is None).")
            return
        phi_2d = phi_variable.value.reshape((nx, ny))
        plt.figure(figsize=(7, 6))
        plt.imshow(phi_2d, origin='lower', cmap='RdBu_r', vmin=0.0, vmax=1.0, interpolation='nearest')
        plt.colorbar(label=r'$\phi$ (Composition)', shrink=0.8) # raw文字列適用
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plot_path = os.path.join(output_dir, "cahn_hilliard_fipy_final_phi.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Cahn-Hilliard plot saved to: {plot_path}")
        plt.show()
        data_path = os.path.join(output_dir, "cahn_hilliard_fipy_final_phi_data.npy")
        np.save(data_path, phi_2d)
        print(f"Cahn-Hilliard data saved to: {data_path}")

    def create_cahn_hilliard_animation(phi_evolution, dt_sim, animation_save_interval_steps, output_dir, filename_base="cahn_hilliard_fipy", fps=20, phi0_mean_val=0.5):
        if not phi_evolution or len(phi_evolution) < 2:
            print("Not enough data for animation (less than 2 frames).")
            return

        print(f"Creating Cahn-Hilliard animation... Number of frames: {len(phi_evolution)}")
        fig, ax = plt.subplots(1,1,figsize=(6,5))
        
        im = ax.imshow(phi_evolution[0], cmap='RdBu_r', vmin=0.0, vmax=1.0, origin='lower', interpolation='nearest')
        cb = fig.colorbar(im, ax=ax, label=r'$\phi(x,y)$', shrink=0.8) # raw文字列適用
        
        sim_time_per_frame = dt_sim * animation_save_interval_steps
        time_text = ax.text(0.65, 0.92, '', transform=ax.transAxes, fontsize=10,
                              bbox=dict(boxstyle="round,pad=0.3", ec='black', fc='white', alpha=0.7))
        ax.set_title(rf'$\phi_0={phi0_mean_val:.2f}$ (FiPy CH)', fontsize=12) # raw文字列適用
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()

        def animate_frame(i):
            im.set_data(phi_evolution[i])
            current_sim_time = sim_time_per_frame * i
            time_text.set_text(f'Time = {current_sim_time:.2f}')
            return [im, time_text]

        ani = animation.FuncAnimation(fig, animate_frame, frames=len(phi_evolution),
                                      interval=max(1, int(1000/fps)), blit=True)

        gif_path = os.path.join(output_dir, f"{filename_base}_phi0-{phi0_mean_val:.2f}.gif")
        try:
            ani.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
            print(f"Animation saved to: {gif_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Ensure Pillow is installed. If issues persist, try `blit=False` in FuncAnimation or check writer availability.")
        plt.close(fig)

# -----------------------------------------------------------------------------
# ■ 2. Flory-Huggins (今回は使用しない)
# -----------------------------------------------------------------------------
def analyze_flory_huggins(N_fh, chi_fh, output_dir):
    pass # 実装は省略

# -----------------------------------------------------------------------------
# ■ 3. HOOMD-blue (今回は使用しない)
# -----------------------------------------------------------------------------
if HOOMD_AVAILABLE:
    def run_hoomd_simulation(num_particles_A, num_particles_B, box_size, sim_steps, dt_md,
                             lj_params_A, lj_params_B, lj_params_AB, output_dir, seed=42):
        pass # 実装は省略
# -----------------------------------------------------------------------------
# ■ メイン実行ブロック
# -----------------------------------------------------------------------------
def main():
    print("シラスガラス薄膜 シミュレーションフレームワーク (FiPy Cahn-Hilliard)")
    print("===================================================================")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output_dir = "out_fipy_ch"
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    print("\n--- Cahn-Hilliard Simulation Parameters (FiPy) ---")
    print(f"  W (double-well): {W_CH}")
    print(f"  Kappa (gradient): {KAPPA}")
    print(f"  Mobility (M): {MOBILITY}")
    print(f"  Grid: {CH_NX}x{CH_NY}, dx: {CH_DX}")
    print(f"  Initial phi_mean: {CH_PHI0_MEAN}, Fluctuation: {CH_PHI0_FLUCTUATION}")
    print(f"  Steps: {CH_STEPS}, dt: {CH_DT}")
    print(f"  Animation save interval (steps): {CH_ANIMATION_SAVE_INTERVAL}")
    print("-------------------------------------------------")

    if FIPY_AVAILABLE:
        print("\n【FiPy Cahn-Hilliard スピノーダル分解シミュレーション】")
        final_phi_ch, phi_evolution = run_cahn_hilliard_simulation_fipy(
            nx=CH_NX, ny=CH_NY, dx=CH_DX,
            phi0_mean=CH_PHI0_MEAN, phi0_fluctuation=CH_PHI0_FLUCTUATION,
            W_param=W_CH, kappa_param=KAPPA, M_param=MOBILITY,
            steps=CH_STEPS, dt_ch=CH_DT,
            animation_save_interval=CH_ANIMATION_SAVE_INTERVAL
        )
        if final_phi_ch is not None:
            visualize_and_save_cahn_hilliard_result(
                final_phi_ch, CH_NX, CH_NY, output_dir,
                title=rf"Final $\phi$ (FiPy, $\phi_0=${CH_PHI0_MEAN:.2f}, t={CH_STEPS*CH_DT:.1f})" # raw文字列適用
            )
            if phi_evolution and len(phi_evolution) >=2 :
                create_cahn_hilliard_animation(
                    phi_evolution,
                    dt_sim=CH_DT,
                    animation_save_interval_steps=CH_ANIMATION_SAVE_INTERVAL,
                    output_dir=output_dir,
                    filename_base="ch_fipy_evolution",
                    fps=CH_ANIMATION_FPS,
                    phi0_mean_val=CH_PHI0_MEAN
                )
            else:
                 print("Skipping animation: Not enough frames in phi_evolution.")
        else:
            print("Cahn-Hilliard simulation (FiPy) did not complete successfully.")
    else:
        print("\nFiPy Cahn-Hilliard スピノーダル分解シミュレーション")
        print("  FiPyが利用できないためスキップします。")
    print("----------------------------------------------")

    print("\n===================================================================")
    print(f"シミュレーションフレームワークの実行が完了しました。結果は {output_dir} に保存されました。")

if __name__ == "__main__":
    main()