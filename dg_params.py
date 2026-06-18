"""
dg_params.py — JEDYNE ŹRÓDŁO PRAWDY dla reżimu częstotliwości wejścia PP.

Powstało, by usunąć niespójność wykrytą przez freq_audit.py (sekcja D):
  - single_neuron_patsep_brian2.py używał 10 Hz/włókno  → 400 Hz aggregate
  - dg_microcircuit_brian2.py / dg_module3 używały 15 Hz/włókno → 600 Hz aggregate

KANON (skalibrowany w freq_audit.py: dla GC Izhikevicza K=10, W_PP_GC=4 daje
output GC ~6 Hz — reżim rzadkiego kodowania DG, aktywne GC ~1–10 Hz):

  • 40 włókien PP na jeden GC
  • aktywny GC:    10 Hz / włókno   → 400 Hz aggregate
  • nieaktywny GC:  1 Hz / włókno   →  40 Hz aggregate

Dlaczego 400, nie 600 Hz?  Przy 600 Hz output GC ≈ 16.6 Hz — za wysoko jak na
DG.  Patrz freq_audit.py oraz Changes.md (Priorytet 1).
"""

# ── Geometria wejścia ─────────────────────────────────────────────────────────
N_FIBERS_PER_GC = 40        # liczba włókien perforant-path na jeden GC

# ── Częstotliwości per-włókno (tryb „jak Madar", biofizycznie wierny) ─────────
R_FIBER_ACTIVE = 10.0       # Hz / włókno — aktywny GC
R_FIBER_BG     =  1.0       # Hz / włókno — tło (nieaktywny GC)

# ── Częstotliwości zagregowane (lumping; superpozycja Poissona) ───────────────
R_EFF_HIGH = N_FIBERS_PER_GC * R_FIBER_ACTIVE   # 400 Hz — aktywny GC
R_EFF_LOW  = N_FIBERS_PER_GC * R_FIBER_BG       #  40 Hz — tło

# ── Waga PP→GC dla neuronu GC Izhikevicza ─────────────────────────────────────
#   g_ex_ss = R_EFF_HIGH · W_PP_GC · τ_ex(5 ms) = 400·4·0.005 = 8 mV
#   (podprogowo względem G_crit≈14 mV przy K=10 → reżim fluktuacyjny → separacja)
W_PP_GC_IZH = 4.0


def aggregate_from_fiber(r_fiber_hz, n_fibers=N_FIBERS_PER_GC):
    """Zamień częstotliwość per-włókno na zagregowaną (superpozycja Poissona)."""
    return r_fiber_hz * n_fibers


def summary():
    """Zwięzły opis kanonu — do wypisania w skryptach."""
    return (f"Wejście PP (kanon dg_params): {N_FIBERS_PER_GC} włókien/GC × "
            f"{R_FIBER_ACTIVE:.0f}/{R_FIBER_BG:.0f} Hz → "
            f"{R_EFF_HIGH:.0f}/{R_EFF_LOW:.0f} Hz aggregate "
            f"(aktywny/tło),  W_PP_GC={W_PP_GC_IZH} mV")


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print(summary())
