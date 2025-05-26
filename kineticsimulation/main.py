import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# シミュレーション設定
# ----------------------------
# プロセスパラメータの範囲
powers = np.linspace(50, 300, 10)         # RF投入電力 [W]
pressures = np.linspace(0.1, 1.0, 10)     # ガス圧力 [Pa]

# 粒子・ガス物性
m_gas = 6.63e-26        # Ar 質量 [kg]
m_tgt = 100 * 1.66e-27  # 平均ターゲット質量 [kg]（例：100 amu）
sigma = 1e-19           # 衝突断面積 [m^2]
k_B = 1.38e-23          # ボルツマン定数
d_sub = 0.05            # ターゲット–基板距離 [m]
N_particles = 5000      # 粒子数（ループごと）

# 初期エネルギー分布のモデル（Thompson 分布の簡易版）
def sample_initial_energy(Eb, Emax=200.0):
    while True:
        E = np.random.rand() * Emax
        if np.random.rand() < E / (E + Eb)**3:
            return E

    # モンテカルロ輸送 → 基板到達時の平均エネルギー
    def mean_energy_at_substrate(power, P_gas):
        # ガス密度と平均自由行程
        
        T_gas = 300.0
        n_gas = P_gas / (k_B * T_gas)
        lambda_mfp = 1.0 / (n_gas * sigma)

        # 投入電力依存のバインディングエネルギー仮定モデル
        Eb0 = 15.0    # 基準バインディングエネルギー [eV]
        P0  = 100.0   # 基準電力 [W]
        Eb = Eb0 * (power / P0)

        final_energies = np.zeros(N_particles)
        for i in range(N_particles):
            E = sample_initial_energy(Eb)
            x = 0.0
            # 伝搬＋衝突ループ
            while x < d_sub and E > 1e-6:
                l = np.random.exponential(lambda_mfp)
                x += l
                if x >= d_sub:
                    break
                alpha = ((m_tgt - m_gas) / (m_tgt + m_gas))**2
                E *= alpha
            final_energies[i] = E

        return final_energies.mean()

# ----------------------------
# 全組み合わせをシミュレーション
# ----------------------------
heatmap = np.zeros((len(pressures), len(powers)))
for i, P_gas in enumerate(pressures):
    for j, Pwr in enumerate(powers):
        heatmap[i, j] = mean_energy_at_substrate(Pwr, P_gas)
        print(f"Power={Pwr:.1f}W, P_gas={P_gas:.2f}Pa → E_mean={heatmap[i,j]:.2f} eV")

# ----------------------------
# ヒートマップのプロット
# ----------------------------
plt.figure(figsize=(8,6))
im = plt.imshow(heatmap,
                origin='lower',
                extent=[powers[0], powers[-1], pressures[0], pressures[-1]],
                aspect='auto',
                cmap='viridis')
cbar = plt.colorbar(im)
cbar.set_label('Mean energy at substrate [eV]')
plt.xlabel('RF Input Power [W]')
plt.ylabel('Argon Pressure [Pa]')
plt.title('Heatmap of Sputtered Particle Energy')
plt.show()
