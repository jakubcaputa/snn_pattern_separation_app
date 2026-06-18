"""
interactive_dg.py — Interaktywna symulacja zakrętu zębatego (Brian2)

Uruchomienie:
    streamlit run interactive_dg.py

Dwa tryby:
  1. Pojedynczy neuron LIF — jeden GC, N skorelowanych wzorców Poissona
  2. Populacja DG (Izhikevich) — pełny obwód GC/FS/HMC, hamowanie FF/FB
"""

import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Brian2 rejestruje SIGINT przy imporcie — patch dla wątków Streamlita
import signal as _signal_mod
_orig_signal = _signal_mod.signal
def _safe_signal(sig, handler):
    try:
        return _orig_signal(sig, handler)
    except ValueError:
        pass
_signal_mod.signal = _safe_signal

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
import time

from brian2 import (
    start_scope, NeuronGroup, Synapses, SpikeMonitor, StateMonitor,
    SpikeGeneratorGroup, Network, defaultclock, ms, mV, prefs,
)
prefs.codegen.target = 'numpy'

st.set_page_config(page_title="DG Microcircuit (Brian2)",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# Wspólne stałe
# ══════════════════════════════════════════════════════════════════════════════

DT_MS    = 0.1     # krok czasowy [ms]
T_MS_LIF = 2000.0  # czas symulacji — tryb pojedynczego neuronu [ms]
T_MS_POP =  600.0  # czas symulacji — tryb populacji [ms]

# Kolory (COL_ prefix — bez kolizji z Izhikeviczem)
COL_GC  = '#1565C0'
COL_FS  = '#C62828'
COL_HMC = '#E65100'
COL_PP  = '#2E7D32'
COL_EX  = '#212121'
COL_IN  = '#6A1B9A'
COL_DIM = '#C8C8C8'

# ══════════════════════════════════════════════════════════════════════════════
# TRYB 1 — Pojedynczy neuron LIF
# ══════════════════════════════════════════════════════════════════════════════

def make_patterns_lif(N, r_hz, R, n_syn, T_ms, dt_ms, seed=42):
    """N wzorców skorelowanych tras Poissona (common + private streams)."""
    rng = np.random.default_rng(seed)
    n_steps   = int(T_ms / dt_ms)
    p_common  = R * r_hz * dt_ms * 1e-3
    p_private = (1.0 - R) * r_hz * dt_ms * 1e-3

    times  = [[] for _ in range(N)]
    binary = [np.zeros((n_syn, n_steps), dtype=bool) for _ in range(N)]

    for s in range(n_syn):
        common = rng.random(n_steps) < p_common
        for k in range(N):
            private  = rng.random(n_steps) < p_private
            combined = common | private
            binary[k][s] = combined
            spk = (np.where(combined)[0] + 1).astype(float) * dt_ms
            times[k].append(spk[spk < T_ms])

    return times, binary


def simulate_lif(spike_times_list, n_syn, W_SYN, tau_m, tau_syn,
                 E_L, V_th, V_reset, t_ref, record_v=False):
    """Jeden neuron LIF, n_syn wejść Poissona."""
    start_scope()
    defaultclock.dt = DT_MS * ms
    T_MS = T_MS_LIF

    lif_eqs = '''
    dv/dt = (E_L_val - v + g) / tau_m_val : volt (unless refractory)
    dg/dt = -g / tau_syn_val              : volt
    '''
    ns = {
        'E_L_val':    E_L    * mV,
        'tau_m_val':  tau_m  * ms,
        'tau_syn_val':tau_syn * ms,
    }

    all_idx = np.concatenate([np.full(len(spk), s, dtype=int)
                               for s, spk in enumerate(spike_times_list)])
    all_t   = np.concatenate(spike_times_list)
    order   = np.argsort(all_t)
    inputs  = SpikeGeneratorGroup(n_syn, all_idx[order], all_t[order] * ms)

    gc = NeuronGroup(1, lif_eqs,
                     threshold=f'v >= {V_th}*mV',
                     reset=f'v = {V_reset}*mV',
                     refractory=t_ref * ms,
                     method='exact',
                     namespace=ns)
    gc.v = E_L * mV
    gc.g = 0 * mV

    syn = Synapses(inputs, gc, on_pre=f'g += {W_SYN}*mV')
    syn.connect()

    sm = SpikeMonitor(gc)
    objs = [inputs, gc, syn, sm]
    if record_v:
        stm = StateMonitor(gc, 'v', record=True)
        objs.append(stm)

    Network(*objs).run(T_MS * ms)

    out_times = np.array(sm.t / ms)
    if record_v:
        return out_times, np.array(stm.t / ms), np.array(stm.v[0] / mV)
    return out_times, None, None


def lif_input_r(binary_list, bin_ms=10.0, T_ms=T_MS_LIF):
    """Średnia parami korelacja zagregowanych wejść."""
    bin_steps = int(bin_ms / DT_MS)
    counts = []
    for p in binary_list:
        agg = p.any(axis=0)
        n   = len(agg) // bin_steps
        counts.append(agg[:n*bin_steps].reshape(n, bin_steps).sum(1).astype(float))
    return _pairwise_r(counts)


def lif_output_r(out_times_list, bin_ms=10.0, T_ms=T_MS_LIF):
    """Średnia parami korelacja wyjściowych spike trainów."""
    bins   = np.arange(0, T_ms + bin_ms, bin_ms)
    counts = [np.histogram(tr, bins=bins)[0].astype(float) for tr in out_times_list]
    return _pairwise_r(counts)


def _pairwise_r(count_list):
    rs = []
    for i in range(len(count_list)):
        for j in range(i + 1, len(count_list)):
            a, b = count_list[i], count_list[j]
            if a.std() > 0 and b.std() > 0:
                r = float(np.corrcoef(a, b)[0, 1])
                if not np.isnan(r):
                    rs.append(r)
    return float(np.mean(rs)) if rs else 0.0


def lif_corr_matrix(times_list, bin_ms=10.0, T_ms=T_MS_LIF):
    bins = np.arange(0, T_ms + bin_ms, bin_ms)
    counts = [np.histogram(tr, bins=bins)[0].astype(float) for tr in times_list]
    N = len(counts)
    R_mat = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            if counts[i].std() > 0 and counts[j].std() > 0:
                r = float(np.corrcoef(counts[i], counts[j])[0, 1])
                if not np.isnan(r):
                    R_mat[i, j] = R_mat[j, i] = r
    return R_mat


# ══════════════════════════════════════════════════════════════════════════════
# TRYB 2 — Populacja DG (Izhikevich)
# ══════════════════════════════════════════════════════════════════════════════

# Parametry Izhikevicza
A_GC, B_GC, C_GC, D_GC     = 0.02, 0.2, -65.0, 6.0
A_FS, B_FS, C_FS, D_FS     = 0.10, 0.2, -65.0, 2.0
A_HMC, B_HMC, C_HMC, D_HMC = 0.02, 0.2, -65.0, 4.0

TAU_EX_GC  = 5.0
TAU_IN_GC  = 8.0
TAU_EX_FS  = 3.0
TAU_EX_HMC = 5.0
PP_DELAY   = 4.0
SYN_DELAY  = 1.0

# Reżim częstotliwości wejścia PP — z jedynego źródła prawdy (dg_params.py).
# Kanon: 400/40 Hz aggregate (= 40 włókien × 10/1 Hz), output GC ~6 Hz.
from dg_params import (
    N_FIBERS_PER_GC, R_FIBER_ACTIVE, R_FIBER_BG, R_EFF_HIGH, R_EFF_LOW,
)

P_PP_FS  = 0.40
P_GC_FS  = 0.40
P_FS_GC  = 0.50
P_GC_HMC = 0.25
P_HMC_FS = 0.40
P_HMC_GC = 0.40


@st.cache_data(show_spinner=False)
def make_connectivity(N_GC, N_FS, N_HMC):
    rng = np.random.default_rng(0)
    def pairs(Ns, Nt, p):
        mask = rng.random((Ns, Nt)) < p
        s, t = np.where(mask)
        return s.astype(np.int32), t.astype(np.int32)
    return {
        'pp_fs' : pairs(N_GC,  N_FS,  P_PP_FS),
        'gc_fs' : pairs(N_GC,  N_FS,  P_GC_FS),
        'fs_gc' : pairs(N_FS,  N_GC,  P_FS_GC),
        'gc_hmc': pairs(N_GC,  N_HMC, P_GC_HMC),
        'hmc_fs': pairs(N_HMC, N_FS,  P_HMC_FS),
        'hmc_gc': pairs(N_HMC, N_GC,  P_HMC_GC),
    }


@st.cache_data(show_spinner=False)
def make_patterns_pop(N_GC, N_patterns, R_in, P_active, seed=42):
    rng = np.random.default_rng(seed)
    common = rng.random(N_GC) < (P_active * R_in)
    pats = np.zeros((N_patterns, N_GC), dtype=bool)
    for k in range(N_patterns):
        pats[k] = common | (rng.random(N_GC) < P_active * (1.0 - R_in))
    rs = []
    for i in range(N_patterns):
        for j in range(i + 1, N_patterns):
            a, b = pats[i].astype(float), pats[j].astype(float)
            if a.std() > 1e-9 and b.std() > 1e-9:
                rs.append(float(np.corrcoef(a, b)[0, 1]))
    return pats, float(np.mean(rs)) if rs else 0.0


@st.cache_data(show_spinner=False)
def make_input_spikes(pattern_active_tuple, seed=1042):
    pattern_active = np.array(pattern_active_tuple)
    rng = np.random.default_rng(seed)
    n_steps = int(T_MS_POP / DT_MS)
    all_idx, all_t = [], []
    for i, active in enumerate(pattern_active):
        rate = R_EFF_HIGH if active else R_EFF_LOW
        p    = rate * DT_MS * 1e-3
        ts   = np.where(rng.random(n_steps) < p)[0].astype(float) * DT_MS
        ts   = ts[(ts > 0) & (ts < T_MS_POP)]
        if len(ts):
            all_idx.append(np.full(len(ts), i, dtype=np.int32))
            all_t.append(ts)
    if all_idx:
        idx = np.concatenate(all_idx)
        t   = np.concatenate(all_t)
        order = np.argsort(t)
        return idx[order], t[order]
    return np.array([], dtype=np.int32), np.array([])


@st.cache_data(show_spinner=False)
def make_input_spikes_per_fiber(pattern_active_tuple, n_syn_pp, r_fiber, r_fiber_bg, seed=1042):
    """
    Per-fiber Poisson (jak Madar): n_syn_pp niezależnych włókien PP na każdy GC.
    Aktywne GC: każde włókno strzela z r_fiber Hz.
    Nieaktywne GC: każde włókno strzela z r_fiber_bg Hz.
    Indeksy w [0, N_GC*n_syn_pp - 1]: włókno f GC i → idx = i*n_syn_pp + f.
    """
    pattern_active = np.array(pattern_active_tuple)
    N_GC = len(pattern_active)
    rng = np.random.default_rng(seed)
    n_steps = int(T_MS_POP / DT_MS)
    all_idx, all_t = [], []
    for i, active in enumerate(pattern_active):
        rate = r_fiber if active else r_fiber_bg
        p = rate * DT_MS * 1e-3
        for f in range(n_syn_pp):
            ts = np.where(rng.random(n_steps) < p)[0].astype(float) * DT_MS
            ts = ts[(ts > 0) & (ts < T_MS_POP)]
            if len(ts):
                all_idx.append(np.full(len(ts), i * n_syn_pp + f, dtype=np.int32))
                all_t.append(ts)
    if all_idx:
        idx = np.concatenate(all_idx)
        t   = np.concatenate(all_t)
        order = np.argsort(t)
        return idx[order], t[order]
    return np.array([], dtype=np.int32), np.array([])


def _izh_eqs(a, b, tau_ex, tau_in=None, K_tonic=0.0, second_ex_tau=None):
    """
    second_ex_tau: jeśli podane, dodaje DRUGI kanał pobudzający g_ex2 (np. dla GC:
    g_ex = PP→GC, g_ex2 = HMC→GC) — pozwala rozbić budżet napięcia na źródła.
    """
    inh_eq   = f"\ndg_in/dt = -g_in / ({tau_in}*ms) : volt" if tau_in else ""
    inh_term = "- g_in/ms " if tau_in else ""
    K_term   = f"- {K_tonic}*mV/ms" if K_tonic != 0.0 else ""
    ex2_eq   = f"\ndg_ex2/dt = -g_ex2 / ({second_ex_tau}*ms) : volt" if second_ex_tau else ""
    ex2_term = "+ g_ex2/ms " if second_ex_tau else ""
    return (
        f"dv/dt  = (0.04/mV/ms * v**2 + 5/ms * v + 140*mV/ms"
        f" - u/ms + g_ex/ms {ex2_term}{inh_term}{K_term}) : volt (unless refractory)\n"
        f"dg_ex/dt = -g_ex / ({tau_ex}*ms) : volt{ex2_eq}{inh_eq}\n"
        f"du/dt  = {a}/ms * ({b} * v - u) : volt\n"
    )


def _syn(src, tgt, si, ti, w_mV, var, delay_ms):
    if len(si) == 0:
        return None
    syn = Synapses(src, tgt, on_pre=f'{var}_post += {w_mV}*mV',
                   delay=delay_ms * ms)
    syn.connect(i=si, j=ti)
    return syn


def simulate_brian(gc_input_idx, gc_input_t_ms, conn, N_GC, N_FS, N_HMC,
                   enable_ff, enable_fb, enable_hmc,
                   W_PP_GC, W_PP_FS, W_GC_FS, W_FS_GC,
                   W_GC_HMC, W_HMC_FS, W_HMC_GC,
                   K_GC, K_FS, K_HMC,
                   per_fiber=False, n_syn_pp=40,
                   budget_active_idx=None, budget_inact_idx=None):
    start_scope()
    defaultclock.dt = DT_MS * ms
    T_MS = T_MS_POP

    # GC: dwa kanały pobudzające — g_ex (PP→GC) i g_ex2 (HMC→GC) — dla rozbicia budżetu
    gc_eqs  = _izh_eqs(A_GC,  B_GC,  TAU_EX_GC,  tau_in=TAU_IN_GC, K_tonic=K_GC,
                       second_ex_tau=TAU_EX_GC)
    # FS: TRZY osobne kanały pobudzające — g_ex (PP→FS), g_ex2 (GC→FS), g_ex3 (HMC→FS)
    # — by zmierzyć przepływ z każdej ścieżki osobno (do schematu).
    fs_eqs  = (
        f"dv/dt = (0.04/mV/ms*v**2 + 5/ms*v + 140*mV/ms - u/ms"
        f" + g_ex/ms + g_ex2/ms + g_ex3/ms - {K_FS}*mV/ms) : volt (unless refractory)\n"
        f"dg_ex/dt  = -g_ex /({TAU_EX_FS}*ms) : volt\n"
        f"dg_ex2/dt = -g_ex2/({TAU_EX_FS}*ms) : volt\n"
        f"dg_ex3/dt = -g_ex3/({TAU_EX_FS}*ms) : volt\n"
        f"du/dt = {A_FS}/ms*({B_FS}*v - u) : volt\n"
    )
    hmc_eqs = _izh_eqs(A_HMC, B_HMC, TAU_EX_HMC, tau_in=None,       K_tonic=K_HMC)

    n_pp_neurons = N_GC * n_syn_pp if per_fiber else N_GC
    pp = SpikeGeneratorGroup(n_pp_neurons, gc_input_idx, gc_input_t_ms * ms)

    gc = NeuronGroup(N_GC, gc_eqs,
                     threshold='v >= 30*mV',
                     reset=f'v = {C_GC}*mV; u = u + {D_GC}*mV',
                     refractory=2*ms, method='euler')
    fs = NeuronGroup(N_FS, fs_eqs,
                     threshold='v >= 30*mV',
                     reset=f'v = {C_FS}*mV; u = u + {D_FS}*mV',
                     refractory=1*ms, method='euler')
    hmc = NeuronGroup(N_HMC, hmc_eqs,
                      threshold='v >= 30*mV',
                      reset=f'v = {C_HMC}*mV; u = u + {D_HMC}*mV',
                      refractory=2*ms, method='euler')

    gc.v  = -70*mV;  gc.u  = B_GC  * (-70*mV);  gc.g_ex  = 0*mV;  gc.g_ex2 = 0*mV;  gc.g_in = 0*mV
    fs.v  = -70*mV;  fs.u  = B_FS  * (-70*mV);  fs.g_ex  = 0*mV;  fs.g_ex2 = 0*mV;  fs.g_ex3 = 0*mV
    hmc.v = -70*mV;  hmc.u = B_HMC * (-70*mV);  hmc.g_ex = 0*mV

    net_objs = [pp, gc, fs, hmc]

    if per_fiber:
        # n_syn_pp niezależnych włókien PP na GC: włókno f GC i → index i*n_syn_pp + f
        pp_gc_src = np.repeat(np.arange(N_GC, dtype=np.int32) * n_syn_pp, n_syn_pp) + \
                    np.tile(np.arange(n_syn_pp, dtype=np.int32), N_GC)
        pp_gc_tgt = np.repeat(np.arange(N_GC, dtype=np.int32), n_syn_pp)
        syn_pp_gc = Synapses(pp, gc, on_pre=f'g_ex_post += {W_PP_GC}*mV',
                             delay=PP_DELAY * ms)
        syn_pp_gc.connect(i=pp_gc_src, j=pp_gc_tgt)
        net_objs.append(syn_pp_gc)
        if enable_ff:
            # PP→FS: użyj włókna 0 każdego GC z istniejącej macierzy połączeń
            pp_fs_src = conn['pp_fs'][0] * n_syn_pp
            s = _syn(pp, fs, pp_fs_src, conn['pp_fs'][1], W_PP_FS, 'g_ex', PP_DELAY)
            if s is not None: net_objs.append(s)
    else:
        syn_pp_gc = Synapses(pp, gc, on_pre=f'g_ex_post += {W_PP_GC}*mV',
                             delay=PP_DELAY * ms)
        syn_pp_gc.connect(i=np.arange(N_GC), j=np.arange(N_GC))
        net_objs.append(syn_pp_gc)
        if enable_ff:
            s = _syn(pp, fs, conn['pp_fs'][0], conn['pp_fs'][1], W_PP_FS, 'g_ex', PP_DELAY)
            if s is not None: net_objs.append(s)

    if enable_fb:
        # GC→FS do osobnego kanału g_ex2 (feedback), by zmierzyć go niezależnie od PP→FS
        s = _syn(gc, fs, conn['gc_fs'][0], conn['gc_fs'][1], W_GC_FS, 'g_ex2', SYN_DELAY)
        if s is not None: net_objs.append(s)
    if enable_ff or enable_fb:
        s = _syn(fs, gc, conn['fs_gc'][0], conn['fs_gc'][1], W_FS_GC, 'g_in', SYN_DELAY)
        if s is not None: net_objs.append(s)
    if enable_hmc:
        s = _syn(gc,  hmc, conn['gc_hmc'][0], conn['gc_hmc'][1], W_GC_HMC, 'g_ex', SYN_DELAY)
        if s is not None: net_objs.append(s)
        # HMC→FS do osobnego kanału g_ex3
        s = _syn(hmc, fs,  conn['hmc_fs'][0], conn['hmc_fs'][1], W_HMC_FS, 'g_ex3', SYN_DELAY)
        if s is not None: net_objs.append(s)
        # HMC→GC trafia do osobnego kanału g_ex2 (re-ekscytacja MC widoczna w budżecie)
        s = _syn(hmc, gc,  conn['hmc_gc'][0], conn['hmc_gc'][1], W_HMC_GC, 'g_ex2', SYN_DELAY)
        if s is not None: net_objs.append(s)

    sm_gc  = SpikeMonitor(gc)
    sm_fs  = SpikeMonitor(fs)
    sm_hmc = SpikeMonitor(hmc)
    net_objs += [sm_gc, sm_fs, sm_hmc]

    # Nagraj przewodności KAŻDEJ ścieżki (osobne kanały) na całych populacjach —
    # z tego liczymy: (1) budżet napięcia GC (przebieg dla aktywnych/nieaktywnych),
    # (2) flow_meas = średni (po populacji i czasie) przepływ z każdego połączenia
    # do schematu — wartości MIERZONE z symulacji, nie ze wzoru.
    budget = None
    flow_meas = None
    record = budget_active_idx is not None and len(budget_active_idx)
    if record:
        stm_gc  = StateMonitor(gc,  ['v', 'g_ex', 'g_ex2', 'g_in'], record=True)
        stm_fs  = StateMonitor(fs,  ['g_ex', 'g_ex2', 'g_ex3'],     record=True)
        stm_hmc = StateMonitor(hmc, ['g_ex'],                       record=True)
        net_objs += [stm_gc, stm_fs, stm_hmc]

    Network(*net_objs).run(T_MS * ms)

    if record:
        act   = np.asarray(budget_active_idx, dtype=int)
        inact = np.asarray([] if budget_inact_idx is None else budget_inact_idx, dtype=int)
        gpp   = np.array(stm_gc.g_ex  / mV)     # PP→GC   (N_GC, n_steps)
        ghmc  = np.array(stm_gc.g_ex2 / mV)     # HMC→GC
        gin   = np.array(stm_gc.g_in  / mV)     # FS→GC
        vrec  = np.array(stm_gc.v     / mV)
        budget = dict(
            t=np.array(stm_gc.t / ms),
            gpp_act=gpp[act].mean(0),  ghmc_act=ghmc[act].mean(0),
            gin_act=gin[act].mean(0),  v_act=vrec[act].mean(0),
            v_inact=(vrec[inact].mean(0) if inact.size else None),
        )
        # flow_meas: średni przepływ [mV] na ścieżkę, MIERZONY z symulacji.
        # Ścieżki → GC liczone po AKTYWNYCH GC (spójnie z budżetem napięcia);
        # ścieżki → FS/HMC po całej populacji celu.
        gfs   = np.array(stm_fs.g_ex  / mV)     # PP→FS
        gfs2  = np.array(stm_fs.g_ex2 / mV)     # GC→FS
        gfs3  = np.array(stm_fs.g_ex3 / mV)     # HMC→FS
        ghmcg = np.array(stm_hmc.g_ex / mV)     # GC→HMC
        _m = lambda a: float(a.mean()) if a.size else 0.0
        flow_meas = {
            'pp_gc': _m(gpp[act]), 'hmc_gc': _m(ghmc[act]), 'fs_gc': _m(gin[act]),
            'pp_fs': _m(gfs), 'gc_fs': _m(gfs2),  'hmc_fs': _m(gfs3),
            'gc_hmc': _m(ghmcg),
        }

    return (np.array(sm_gc.i),  np.array(sm_gc.t  / ms),
            np.array(sm_fs.i),  np.array(sm_fs.t  / ms),
            np.array(sm_hmc.i), np.array(sm_hmc.t / ms),
            budget, flow_meas)


def bin_spikes(i_arr, t_arr, N, bin_ms=100.0):
    n_bins = max(1, int(T_MS_POP / bin_ms))
    mat = np.zeros((N, n_bins))
    if len(t_arr):
        bi = np.clip((t_arr / bin_ms).astype(int), 0, n_bins - 1)
        for ci, b in zip(i_arr, bi):
            mat[ci, b] += 1
    return mat


def mean_pairwise_r(binned_list):
    rs = []
    for i in range(len(binned_list)):
        for j in range(i + 1, len(binned_list)):
            a, b = binned_list[i].ravel(), binned_list[j].ravel()
            if a.std() > 1e-9 and b.std() > 1e-9:
                r = float(np.corrcoef(a, b)[0, 1])
                if not np.isnan(r): rs.append(r)
    return float(np.mean(rs)) if rs else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Schemat obwodu (tylko tryb populacja)
# ══════════════════════════════════════════════════════════════════════════════

CPOS = {'PP': np.array([1.8,5.0]), 'GC': np.array([5.0,5.0]),
        'FS': np.array([7.6,7.6]), 'HMC': np.array([7.6,2.4])}
CRAD = {'PP':0.68, 'GC':1.00, 'FS':0.72, 'HMC':0.62}
CFC  = {'PP':COL_PP, 'GC':COL_GC, 'FS':COL_FS, 'HMC':COL_HMC}
# crv = krzywizna łuku; pary dwukierunkowe (GC↔FS, GC↔HMC) mają przeciwne znaki
# i dużą wartość, by były to DWA wyraźnie rozdzielone łuki, nie jedna strzałka.
CCONN = [
    ('pp_gc','PP','GC','ex', 0.00), ('pp_fs','PP','FS','ex', 0.15),
    ('gc_fs','GC','FS','ex', 0.40), ('fs_gc','FS','GC','in',-0.40),
    ('gc_hmc','GC','HMC','ex',0.40),('hmc_fs','HMC','FS','ex',0.00),
    ('hmc_gc','HMC','GC','ex',-0.40),
]
CONN_LABELS = {
    'pp_gc' :('PP→GC',   (3.4,5.75)), 'pp_fs' :('PP→FS',   (3.7,7.10)),
    'gc_fs' :('GC→FS',   (6.95,6.85)),'fs_gc' :('FS→GC',   (5.55,5.55)),
    'gc_hmc':('GC→HMC',  (7.15,3.35)),'hmc_fs':('HMC→FS',  (8.30,5.00)),
    'hmc_gc':('HMC→GC',  (5.45,3.25)),
}

def _cactive(cid, ff, fb, hmc):
    if cid == 'pp_gc': return True
    if cid == 'pp_fs': return ff
    if cid == 'gc_fs': return fb
    if cid == 'fs_gc': return ff or fb
    return hmc


def compute_flows(res):
    """
    Przepływ ładunku przez każde połączenie — wartości MIERZONE z symulacji:
    średnia (po komórkach-celach i po czasie) przewodności dostarczonej przez
    daną ścieżkę (`flow_meas` z simulate_brian; osobny kanał syn. na ścieżkę).
    Zwraca (val, lw): słownik wartości [mV] + słownik grubości linii (1–5.5,
    kompresja sqrt, by słabe przepływy pozostały widoczne).
    """
    val = res.get('flow_meas') or {}
    mx = max(val.values()) if val and max(val.values()) > 0 else 1.0
    lw = {cid: 1.0 + 4.5 * (v / mx) ** 0.5 for cid, v in val.items() if v > 0}
    return val, lw


def draw_circuit(ax, ff, fb, hmc, N_GC, N_FS, N_HMC, flow_val=None, flow_lw=None):
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_aspect('equal'); ax.axis('off')
    counts = {'PP':'40 wł.','GC':f'N={N_GC}','FS':f'N={N_FS}','HMC':f'N={N_HMC}'}
    labels = {'PP':'PP\n(wejście)','GC':'GC','FS':'FS','HMC':'HMC'}
    for n, xy in CPOS.items():
        ax.add_patch(Circle(tuple(xy), CRAD[n], fc=CFC[n], ec='black', lw=1.3, zorder=5))
        ax.text(*xy, f"{labels[n]}\n{counts[n]}", ha='center', va='center',
                fontsize=6.5, fontweight='bold', color='white', zorder=6, multialignment='center')
    for cid, src, dst, stype, crv in CCONN:
        active = _cactive(cid, ff, fb, hmc)
        p1, p2 = CPOS[src], CPOS[dst]
        u = (p2-p1)/np.linalg.norm(p2-p1)
        perp = np.array([-u[1], u[0]])      # wektor prostopadły do linii połączenia
        # KAŻDE połączenie = osobna strzałka jednokierunkowa (-> pobudzenie, -[ hamowanie).
        # Pary dwukierunkowe (GC↔FS, GC↔HMC) rozsuwamy na DWA równoległe pasy: offset
        # liczony z kierunku src→dst, więc A→B i B→A trafiają po przeciwnych stronach.
        OFF = 0.26 if crv != 0 else 0.0
        start = tuple(p1 + CRAD[src]*u + OFF*perp)
        end   = tuple(p2 - CRAD[dst]*u + OFF*perp)
        color = (COL_EX if stype=='ex' else COL_IN) if active else COL_DIM
        lw = (flow_lw.get(cid, 1.8) if flow_lw else 1.8) if active else 0.8
        ax.add_patch(FancyArrowPatch(start, end,
            arrowstyle='->' if stype=='ex' else '-[', color=color,
            linewidth=lw, alpha=1.0 if active else 0.30,
            connectionstyle=f'arc3,rad={0.12*np.sign(crv)}', mutation_scale=13, zorder=4))
        if cid in CONN_LABELS:
            lbl,(tx,ty) = CONN_LABELS[cid]
            # Dopisz wartość przepływu ładunku [mV] ze znakiem (+ pobudza, − hamuje)
            if active and flow_val and cid in flow_val:
                sign = '−' if stype == 'in' else '+'
                lbl = f"{lbl}\n{sign}{flow_val[cid]:.1f} mV"
            ax.text(tx,ty,lbl, fontsize=4.6, ha='center', va='center', color=color, zorder=7,
                    fontweight='bold' if active else 'normal',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, lw=0.4,
                              alpha=0.90 if active else 0.35))


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("Interaktywna Symulacja Zakrętu Zębatego (Brian2)")
st.caption(f"LIF: T_sim={T_MS_LIF:.0f} ms · Populacja: T_sim={T_MS_POP:.0f} ms · dt={DT_MS} ms · Brian2")

# ── Selektor trybu (na górze sidebara) ───────────────────────────────────────
with st.sidebar:
    mode = st.radio("Tryb symulacji",
                    ["🧠 Pojedynczy neuron (LIF)", "🌐 Populacja DG (Izhikevich)"],
                    key="mode")
    st.markdown("---")

    # ── Kontrolki LIF ─────────────────────────────────────────────────────────
    if mode == "🧠 Pojedynczy neuron (LIF)":
        st.header("Parametry neuronu LIF")
        lif_tau_m   = st.slider("τ_m  (czas membranowy, ms)",   5.0, 60.0, 20.0, step=1.0)
        lif_V_th    = st.slider("V_th  (próg, mV)",            -65.0,-45.0,-55.0, step=1.0)
        lif_V_reset = st.slider("V_reset  (reset, mV)",        -85.0,-60.0,-78.0, step=1.0)
        lif_t_ref   = st.slider("t_ref  (refrakcja, ms)",        1.0,  8.0,  3.0, step=0.5)

        st.markdown("---")
        st.header("Synapsy wejściowe")
        lif_n_syn   = st.slider("n_syn  (liczba synaps PP)",   10, 100,  40, step=5)
        lif_W_SYN   = st.slider("W_syn  (waga synaptyczna, mV)", 0.5, 20.0, 6.0, step=0.5)
        lif_tau_syn = st.slider("τ_syn  (czas synaptyczny, ms)", 1.0, 20.0, 5.0, step=0.5)
        lif_r_input = st.slider("r_input  (częst. wejścia, Hz)", 2.0, 50.0,10.0, step=1.0)

        st.markdown("---")
        st.header("Wzorce wejściowe")
        lif_R_in       = st.slider("R_in  (korelacja wzorców)",  0.10, 0.98, 0.80, step=0.05)
        lif_N_patterns = st.slider("Liczba wzorców", 2, 6, 5, step=1)

    # ── Kontrolki populacji ───────────────────────────────────────────────────
    else:
        st.header("Sieć neuronowa")
        N_GC  = st.slider("N_GC (granule cells)",      50, 400, 200, step=50)
        N_FS  = st.slider("N_FS (interneurony FS)",     5,  40,  20, step=5)
        N_HMC = st.slider("N_HMC (hilar mossy cells)", 2,  20,  10, step=2)

        st.markdown("---")
        st.header("Aktywne połączenia")
        enable_ff  = st.checkbox("Feedforward  (PP→FS→GC)", value=True)
        enable_fb  = st.checkbox("Feedback     (GC→FS→GC)", value=True)
        enable_hmc = st.checkbox("Hilar Mossy Cells (HMC)", value=True)

        st.markdown("---")
        st.header("Wagi synaptyczne [mV]")
        W_PP_GC = st.slider("W PP→GC  (wejście PP)",  0.5, 12.0,  4.0, step=0.5)
        W_PP_FS = st.slider("W PP→FS  (siła FF)",     0.05, 2.0,  0.25, step=0.05)
        W_GC_FS = st.slider("W GC→FS  (siła FB)",     1.0, 30.0, 10.0, step=1.0)
        W_FS_GC = st.slider("W FS→GC  (hamowanie)",   0.1,  5.0,  1.0, step=0.1)
        with st.expander("Wagi Mossy Cells (HMC)", expanded=True):
            st.caption("Domyślnie HMC są prawie ciche: rzadko strzelające GC dają zbyt "
                       "mały napęd **W GC→HMC** względem progu (G_crit = 4 + K_HMC = 14 mV). "
                       "Aby je aktywować, zwiększ **W GC→HMC** (≈10–16) lub zmniejsz "
                       "**K_HMC**; wtedy w budżecie napięcia urośnie pomarańczowe pole "
                       "**W HMC→GC** (re-ekscytacja MC).")
            W_GC_HMC = st.slider("W GC→HMC  (napęd MC)",        0.5, 20.0,  1.0, step=0.5)
            W_HMC_FS = st.slider("W HMC→FS  (MC wzmacnia FS)",  0.0, 10.0,  1.0, step=0.5)
            W_HMC_GC = st.slider("W HMC→GC  (re-ekscytacja MC)", 0.0,  5.0,  0.5, step=0.1)

        st.markdown("---")
        st.header("Parametry wejściowe")
        R_in       = st.slider("R_in  (podobieństwo wzorców)", 0.20, 0.98, 0.75, step=0.05)
        P_active   = st.slider("P_active  (% aktywnych GC)",   0.10, 0.50, 0.25, step=0.05)
        N_patterns = st.slider("Liczba wzorców",                2,    5,    3,    step=1)

        st.markdown("---")
        st.header("Model sygnału PP")
        input_mode = st.radio(
            "Typ wejścia PP",
            ["Zagregowany (szybszy)", "Per-fiber (jak Madar)"],
            help="Per-fiber: każdy GC dostaje n_syn_pp niezależnych włókien PP ~10 Hz, "
                 "jak w pracy Madara. Wolniejszy, ale biofizycznie wierniejszy.")
        per_fiber = (input_mode == "Per-fiber (jak Madar)")
        if per_fiber:
            n_syn_pp   = st.slider("n_syn_pp  (włókna PP / GC)", 5, 80, N_FIBERS_PER_GC, step=5)
            r_fiber    = st.slider("r_fiber  (Hz / włókno, aktywne GC)", 1.0, 30.0, R_FIBER_ACTIVE, step=1.0)
            r_fiber_bg = st.slider("r_fiber_bg  (Hz / włókno, nieaktywne GC)", 0.1, 5.0, R_FIBER_BG, step=0.1)
            st.caption(f"Łączna częst. aktywnego GC: **{n_syn_pp * r_fiber:.0f} Hz** "
                       f"(vs zagregowane {R_EFF_HIGH:.0f} Hz). "
                       f"Zalecane W PP→GC ≈ {R_EFF_HIGH / (n_syn_pp * r_fiber) * 4:.1f} mV "
                       f"aby zachować ten sam drive.")
        else:
            n_syn_pp   = 1
            r_fiber    = R_EFF_HIGH
            r_fiber_bg = R_EFF_LOW

        st.markdown("---")
        st.header("Pobudliwość (K_tonic)")
        K_GC  = st.slider("K_tonic GC",  0.0, 20.0, 10.0, step=1.0)
        K_FS  = st.slider("K_tonic FS",  0.0, 15.0,  5.0, step=1.0)
        K_HMC = st.slider("K_tonic HMC", 0.0, 20.0, 10.0, step=1.0)

    st.markdown("---")
    run_btn = st.button("▶ Uruchom symulację", type="primary", use_container_width=True)

# ── Klucz parametrów ──────────────────────────────────────────────────────────
if mode == "🧠 Pojedynczy neuron (LIF)":
    params_key = ("lif", round(lif_tau_m,1), round(lif_V_th,1), round(lif_V_reset,1),
                  round(lif_t_ref,1), lif_n_syn, round(lif_W_SYN,1), round(lif_tau_syn,1),
                  round(lif_r_input,1), round(lif_R_in,2), lif_N_patterns)
else:
    params_key = ("pop", N_GC, N_FS, N_HMC, enable_ff, enable_fb, enable_hmc,
                  round(W_PP_GC,2), round(W_PP_FS,3), round(W_GC_FS,1), round(W_FS_GC,2),
                  round(W_GC_HMC,2), round(W_HMC_FS,2), round(W_HMC_GC,2),
                  round(R_in,2), round(P_active,2), N_patterns,
                  round(K_GC,1), round(K_FS,1), round(K_HMC,1),
                  per_fiber, n_syn_pp if per_fiber else 0,
                  round(r_fiber, 1) if per_fiber else 0,
                  round(r_fiber_bg, 2) if per_fiber else 0)

if "results" not in st.session_state:
    st.session_state.results  = None
    st.session_state.last_key = None

# ══════════════════════════════════════════════════════════════════════════════
# SYMULACJA
# ══════════════════════════════════════════════════════════════════════════════

if run_btn or st.session_state.last_key != params_key:
    st.session_state.last_key = params_key
    prog = st.progress(0, text="Symulacja Brian2…")
    t0   = time.time()

    # ── LIF ──────────────────────────────────────────────────────────────────
    if mode == "🧠 Pojedynczy neuron (LIF)":
        pat_times, pat_bin = make_patterns_lif(
            lif_N_patterns, lif_r_input, lif_R_in, lif_n_syn, T_MS_LIF, DT_MS)
        r_in_actual = lif_input_r(pat_bin)

        out_times_list = []
        vtraces        = []
        for k in range(lif_N_patterns):
            prog.progress((k+0.5)/lif_N_patterns, text=f"Wzorzec {k+1}/{lif_N_patterns}…")
            rec = (k < 2)  # nagrywaj napięcie dla wzorców 1 i 2
            ot, vt, vm = simulate_lif(
                pat_times[k], lif_n_syn, lif_W_SYN, lif_tau_m, lif_tau_syn,
                -70.0, lif_V_th, lif_V_reset, lif_t_ref, record_v=rec)
            out_times_list.append(ot)
            if rec:
                vtraces.append((vt, vm))

        r_out_actual = lif_output_r(out_times_list)
        fr_mean = np.mean([len(t)/(T_MS_LIF*1e-3) for t in out_times_list])
        prog.empty()

        st.session_state.results = dict(
            mode='lif',
            pat_bin=pat_bin, out_times=out_times_list, vtraces=vtraces,
            r_in=r_in_actual, r_out=r_out_actual, dec=r_in_actual-r_out_actual,
            fr=fr_mean, elapsed=time.time()-t0,
            N_patterns=lif_N_patterns, V_th=lif_V_th, E_L=-70.0,
        )

    # ── Populacja ─────────────────────────────────────────────────────────────
    else:
        conn = make_connectivity(N_GC, N_FS, N_HMC)
        pats, r_in_actual = make_patterns_pop(N_GC, N_patterns, R_in, P_active)
        gc_sp, fs_sp, hmc_sp = [], [], []
        budget0 = None     # budżet napięcia dla wzorca 1
        flow_meas0 = None  # zmierzone przepływy (do schematu) dla wzorca 1

        for k in range(N_patterns):
            prog.progress(k/N_patterns, text=f"Wzorzec {k+1}/{N_patterns}…")
            if per_fiber:
                idx, t_ms = make_input_spikes_per_fiber(
                    tuple(pats[k].tolist()), n_syn_pp, r_fiber, r_fiber_bg, seed=1042+k)
            else:
                idx, t_ms = make_input_spikes(tuple(pats[k].tolist()), seed=1042+k)

            # Dla 1. wzorca nagraj budżet napięcia ≤30 aktywnych i ≤30 nieaktywnych GC
            if k == 0:
                act   = np.where(pats[0])[0][:30].astype(int)
                inact = np.where(~pats[0])[0][:30].astype(int)
            else:
                act = inact = None

            gc_i,gc_t,fs_i,fs_t,hmc_i,hmc_t,budget,flow_meas = simulate_brian(
                idx, t_ms, conn, N_GC, N_FS, N_HMC,
                enable_ff, enable_fb, enable_hmc,
                W_PP_GC, W_PP_FS, W_GC_FS, W_FS_GC,
                W_GC_HMC, W_HMC_FS, W_HMC_GC, K_GC, K_FS, K_HMC,
                per_fiber=per_fiber, n_syn_pp=n_syn_pp,
                budget_active_idx=act, budget_inact_idx=inact)
            if k == 0:
                budget0 = budget
                flow_meas0 = flow_meas
            gc_sp.append((gc_i,gc_t)); fs_sp.append((fs_i,fs_t)); hmc_sp.append((hmc_i,hmc_t))

        prog.progress(1.0, text="Obliczanie metryk…")
        binned_gc = [bin_spikes(i,t,N_GC,100.0) for i,t in gc_sp]
        r_out = mean_pairwise_r(binned_gc)
        prog.empty()

        st.session_state.results = dict(
            mode='pop',
            gc_spikes=gc_sp, fs_spikes=fs_sp, hmc_spikes=hmc_sp,
            fr_gc =np.mean([len(t)/(T_MS_POP*1e-3*N_GC)  for _,t in gc_sp]),
            fr_fs =np.mean([len(t)/(T_MS_POP*1e-3*N_FS)  for _,t in fs_sp]),
            fr_hmc=np.mean([len(t)/(T_MS_POP*1e-3*N_HMC) for _,t in hmc_sp]),
            r_in=r_in_actual, r_out=r_out, dec=r_in_actual-r_out,
            elapsed=time.time()-t0,
            N_GC=N_GC, N_FS=N_FS, N_HMC=N_HMC, N_patterns=N_patterns,
            enable_ff=enable_ff, enable_fb=enable_fb, enable_hmc=enable_hmc,
            per_fiber=per_fiber, n_syn_pp=n_syn_pp,
            r_fiber=r_fiber, r_fiber_bg=r_fiber_bg,
            budget=budget0, flow_meas=flow_meas0,
            K_GC=K_GC, K_FS=K_FS, K_HMC=K_HMC,
        )

# ══════════════════════════════════════════════════════════════════════════════
# WIZUALIZACJA
# ══════════════════════════════════════════════════════════════════════════════

res = st.session_state.results

if res is None:
    st.info("Kliknij ▶ Uruchom symulację lub zmień parametr.")
    st.stop()

PAT_COLORS = plt.cm.tab10(np.linspace(0, 0.5, res['N_patterns']))

# ════════════════════ TRYB LIF ════════════════════════════════════════════════
if res['mode'] == 'lif':
    # Metryki
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FR GC (LIF)",     f"{res['fr']:.1f} Hz")
    m2.metric("R_in",            f"{res['r_in']:.3f}")
    m3.metric("R_out",           f"{res['r_out']:.3f}")
    m4.metric("Dekorelajcja",    f"{res['dec']:+.3f}",
              delta=f"{res['elapsed']:.1f}s")

    col_l, col_r = st.columns(2)

    # Input raster
    with col_l:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for k, pb in enumerate(res['pat_bin']):
            agg_t = np.where(pb.any(axis=0))[0] * DT_MS
            ax.scatter(agg_t, np.full_like(agg_t, k+1, dtype=float),
                       s=1.0, color=PAT_COLORS[k], alpha=0.6, linewidths=0)
        ax.set(xlim=(0,T_MS_LIF), ylim=(0.3, res['N_patterns']+0.7),
               xlabel='Czas (ms)', ylabel='Wzorzec #',
               title=f'Wejście PP — zagregowany spike train\n(R_in = {res["r_in"]:.3f})')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # Output raster
    with col_r:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for k, spk in enumerate(res['out_times']):
            ax.scatter(spk, np.full_like(spk, k+1),
                       s=8, color=PAT_COLORS[k], alpha=0.9, linewidths=0)
        ax.set(xlim=(0,T_MS_LIF), ylim=(0.3, res['N_patterns']+0.7),
               xlabel='Czas (ms)', ylabel='Wzorzec #',
               title=f'Wyjście GC (LIF) — spike train\n(R_out = {res["r_out"]:.3f}  |  dec = {res["dec"]:+.3f})')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # Voltage traces (wzorce 1 i 2)
    st.subheader("Potencjał błonowy (wzorce 1 i 2)")
    fig, ax = plt.subplots(figsize=(11, 3))
    for k, (vt, vm) in enumerate(res['vtraces']):
        ax.plot(vt, vm, lw=0.9, alpha=0.9,
                color=PAT_COLORS[k], label=f'Wzorzec {k+1}')
    ax.axhline(res['V_th'], color='black', ls='--', lw=1.0, alpha=0.7,
               label=f'V_th = {res["V_th"]} mV')
    ax.axhline(res['E_L'],  color='gray',  ls=':',  lw=0.8, alpha=0.5,
               label=f'E_L = {res["E_L"]} mV')
    ax.set(xlim=(0,T_MS_LIF), xlabel='Czas (ms)', ylabel='V_m (mV)',
           title='Pomimo podobnych wejść (R_in > 0), wzorce generują różne czasy spajków')
    ax.legend(fontsize=8, loc='upper right', ncol=4)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # Macierze korelacji
    st.subheader("Macierze korelacji (input vs output)")
    col_a, col_b = st.columns(2)
    for col, title, times_list, is_input in [
        (col_a, f'Input  (mean R = {res["r_in"]:.3f})',
         [np.where(res['pat_bin'][k].any(axis=0))[0]*DT_MS for k in range(res['N_patterns'])], True),
        (col_b, f'Output  (mean R = {res["r_out"]:.3f})', res['out_times'], False),
    ]:
        R_mat = lif_corr_matrix(times_list)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(R_mat, vmin=-0.2, vmax=1.0, cmap='RdYlGn', aspect='auto')
        plt.colorbar(im, ax=ax, label='Pearson R', shrink=0.8)
        ax.set(xlabel='Wzorzec #', ylabel='Wzorzec #', title=title)
        fig.tight_layout(); col.pyplot(fig, use_container_width=True); plt.close(fig)

# ════════════════════ TRYB POPULACJA ══════════════════════════════════════════
else:
    N_GC_r = res['N_GC']; N_FS_r = res['N_FS']; N_HMC_r = res['N_HMC']

    col_circ, col_metrics = st.columns([1.1, 1.9])

    with col_circ:
        st.subheader("Schemat obwodu")
        flow_val, flow_lw = compute_flows(res)
        fig_c, ax_c = plt.subplots(figsize=(4, 4))
        draw_circuit(ax_c, res['enable_ff'], res['enable_fb'], res['enable_hmc'],
                     N_GC_r, N_FS_r, N_HMC_r, flow_val=flow_val, flow_lw=flow_lw)
        ax_c.legend(handles=[
            mpatches.Patch(color=COL_EX, label='Pobudzające (AMPA)'),
            mpatches.Patch(color=COL_IN, label='Hamujące (GABA-A)'),
            mpatches.Patch(color=COL_DIM, label='Nieaktywne'),
        ], fontsize=5.5, loc='lower left', framealpha=0.9)
        fig_c.tight_layout(); st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)
        st.caption("Każda strzałka = jeden kierunek (→ pobudza, ⊣ hamuje); połączenia "
                   "dwukierunkowe (GC↔FS, GC↔HMC) to dwie osobne strzałki. Grubość oraz "
                   "liczba [mV] = przepływ ZMIERZONY z symulacji (średnia przewodności w "
                   "czasie; ścieżki→GC po aktywnych GC, spójnie z budżetem; →FS/HMC po "
                   "całej populacji). Znak +/− = pobudza/hamuje.")

        cname = ("full"     if res['enable_ff'] and res['enable_fb']
                 else "ff_only" if res['enable_ff']
                 else "fb_only" if res['enable_fb'] else "baseline")
        cc = {'baseline':'#9E9E9E','ff_only':'#1976D2','fb_only':'#D32F2F','full':'#2E7D32'}
        cl = {'baseline':'Baseline','ff_only':'Feedforward PP→FS→GC',
              'fb_only':'Feedback GC→FS→GC','full':'Pełny obwód (FF+FB)'}
        st.markdown(f"<div style='background:{cc[cname]};color:white;padding:6px 12px;"
                    f"border-radius:6px;text-align:center;font-weight:bold'>{cl[cname]}</div>",
                    unsafe_allow_html=True)

    with col_metrics:
        st.subheader("Wyniki")
        if res['per_fiber']:
            st.info(f"**Model PP: per-fiber** — {res['n_syn_pp']} włókien/GC × "
                    f"{res['r_fiber']:.0f} Hz (aktywne) / {res['r_fiber_bg']:.1f} Hz (nieaktywne)")
        else:
            st.info(f"**Model PP: zagregowany** — {R_EFF_HIGH:.0f} Hz (aktywne) / "
                    f"{R_EFF_LOW:.0f} Hz (nieaktywne)")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("FR GC",  f"{res['fr_gc']:.1f} Hz",
                  delta=("rzadkie ✓" if res['fr_gc'] <= 10 else "za gęste ✗"),
                  delta_color="off",
                  help="DG koduje rzadko: aktywne GC ~1–10 Hz.")
        m2.metric("FR FS",  f"{res['fr_fs']:.1f} Hz",
                  delta=("gamma ✓" if 30 <= res['fr_fs'] <= 100
                         else ("za wolno" if res['fr_fs'] < 30 else "za szybko")),
                  delta_color="off",
                  help="Interneurony FS powinny strzelać w paśmie gamma 30–100 Hz.")
        m3.metric("FR HMC", f"{res['fr_hmc']:.1f} Hz",
                  delta=("aktywne" if res['fr_hmc'] >= 0.5 else "~ciche"),
                  delta_color="off",
                  help="Mossy cells — w domyślnym setupie często prawie nieaktywne.")
        m4.metric("R_in → R_out", f"{res['r_in']:.2f} → {res['r_out']:.2f}")
        m5.metric("Dekorelacja", f"{res['dec']:+.3f}")
        m6.metric("Czas", f"{res['elapsed']:.1f}s")

        # Raster plots — N_patterns wzorców UŁOŻONYCH PIONOWO (każdy w osobnym pasmie).
        # Oś Y pokazuje numer wzorca (Wz.k), NIE surowy indeks neuronu, bo pasma są
        # przesunięte o offset (inaczej oś sięgałaby N_patterns × N_neuronów).
        fig_r, axes = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True,
                                   gridspec_kw={'hspace':0.18})

        def _draw_raster(ax, spikes, N_cells, gap, title, size, legend=False):
            band = N_cells + gap
            for k, (ci, ct) in enumerate(spikes):
                ax.scatter(ct, ci + k * band, s=size, color=PAT_COLORS[k],
                           alpha=0.75, linewidths=0,
                           label=f'Wzorzec {k+1}' if legend else None)
                if k > 0:                                   # linia rozdzielająca pasma
                    ax.axhline(k * band - gap / 2, color='lightgray', lw=0.6, zorder=0)
            ax.set_ylim(-gap, res['N_patterns'] * band)
            ax.set_yticks([k * band + N_cells / 2 for k in range(res['N_patterns'])])
            ax.set_yticklabels([f'Wz.{k+1}\n(0–{N_cells})' for k in range(res['N_patterns'])],
                               fontsize=6)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            if legend:
                ax.legend(loc='upper right', fontsize=6.5, markerscale=5)

        _draw_raster(axes[0], res['gc_spikes'],  N_GC_r,  6,
                     f"Raster — Granule Cells (GC), N={N_GC_r}",  0.8, legend=True)
        _draw_raster(axes[1], res['fs_spikes'],  N_FS_r,  3,
                     f"Raster — FS Interneurons, N={N_FS_r}",     1.5)
        _draw_raster(axes[2], res['hmc_spikes'], N_HMC_r, 2,
                     f"Raster — Hilar Mossy Cells (HMC), N={N_HMC_r}", 1.5)
        axes[2].set_xlabel("Czas (ms)", fontsize=8)
        axes[2].set_xlim(0, T_MS_POP)
        # Adnotacja gdy HMC milczą — żeby pusty panel nie mylił
        if res['fr_hmc'] < 0.05:
            axes[2].text(0.5, 0.5, 'HMC nieaktywne (0 Hz) — patrz panel „Rola Mossy Cells”',
                         transform=axes[2].transAxes, ha='center', va='center',
                         fontsize=8, color='gray', style='italic')
        fig_r.tight_layout(); st.pyplot(fig_r, use_container_width=True); plt.close(fig_r)

    # ── Budżet napięcia GC: kto pobudza, a kto hamuje ─────────────────────────
    st.subheader("Co pobudza, a co hamuje GC?  (budżet napięcia)")
    bud = res.get('budget')
    if bud is None:
        st.caption("Brak danych budżetu napięcia (żaden GC nie był aktywny we wzorcu 1).")
    else:
        G_crit = 4.0 + res['K_GC']
        fig_b, (axb1, axb2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                                           gridspec_kw={'hspace':0.12})
        t    = bud['t']
        gpp  = bud['gpp_act']; ghmc = bud['ghmc_act']; gin = bud['gin_act']
        net  = gpp + ghmc - gin
        # Pobudzenie ułożone warstwowo: PP (zielony) + HMC→GC (pomarańcz) ku górze
        axb1.fill_between(t, 0, gpp, color=COL_PP, alpha=0.40,
                          label='Pobudzenie PP→GC  (+g_ex)')
        axb1.fill_between(t, gpp, gpp + ghmc, color=COL_HMC, alpha=0.55,
                          label='Pobudzenie HMC→GC  (+g_ex2, re-ekscytacja MC)')
        axb1.fill_between(t, 0, -gin, color=COL_IN, alpha=0.35,
                          label='Hamowanie FS→GC  (−g_in)')
        axb1.plot(t, net, color=COL_EX, lw=1.1, label='Napęd netto (PP+HMC−FS)')
        axb1.axhline(G_crit, color='crimson', ls='--', lw=1.0,
                     label=f'G_crit ≈ {G_crit:.0f} mV (próg, K_GC={res["K_GC"]:.0f})')
        axb1.axhline(0, color='gray', lw=0.6)
        axb1.set_ylabel('Wkład do V [mV]', fontsize=8)
        axb1.set_title('Uśredniony wkład synaptyczny wg ŹRÓDŁA — aktywne GC (wzorzec 1)',
                       fontsize=9, fontweight='bold')
        axb1.legend(fontsize=6.0, loc='upper right', ncol=2)
        axb1.spines[['top','right']].set_visible(False)

        axb2.plot(t, bud['v_act'], color=COL_GC, lw=0.7, label='V_m aktywne GC (śred.)')
        if bud.get('v_inact') is not None:
            axb2.plot(t, bud['v_inact'], color=COL_DIM, lw=0.7, label='V_m nieaktywne GC (śred.)')
        axb2.set_ylabel('V_m [mV]', fontsize=8); axb2.set_xlabel('Czas (ms)', fontsize=8)
        axb2.set_xlim(0, T_MS_POP)
        axb2.legend(fontsize=6.5, loc='upper right')
        axb2.spines[['top','right']].set_visible(False)
        fig_b.tight_layout(); st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)
        st.caption("Zielone = pobudzenie PP→GC; pomarańczowe = re-ekscytacja HMC→GC "
                   "(jeśli pole jest cienkie, mossy cells prawie nie dokładają); "
                   "fioletowe = hamowanie FS→GC. Czarna linia = napęd netto; im częściej "
                   "przebija G_crit, tym częściej GC odpala. Wyłącz FF/FB lub HMC w panelu "
                   "bocznym, by zobaczyć wkład każdego źródła z osobna.")

        # Diagnoza roli mossy cells (P4)
        ghmc_mean = float(np.mean(bud['ghmc_act']))
        if not res['enable_hmc']:
            mc_msg = "HMC wyłączone (odznaczone w panelu bocznym)."
        elif res['fr_hmc'] < 0.5:
            mc_msg = (f"HMC praktycznie **ciche** ({res['fr_hmc']:.2f} Hz) → re-ekscytacja "
                      f"HMC→GC ≈ {ghmc_mean:.2f} mV (znikoma). Aby je ożywić: zwiększ "
                      f"**W GC→HMC** lub zmniejsz **K_HMC** (próg = 4+K_HMC).")
        else:
            mc_msg = (f"HMC **aktywne** ({res['fr_hmc']:.1f} Hz) → re-ekscytacja HMC→GC ≈ "
                      f"{ghmc_mean:.2f} mV (pomarańczowe pole); równolegle HMC→FS wzmacnia "
                      f"hamowanie. Netto efekt MC widać porównując R_out z/bez HMC.")
        st.info("**Rola Mossy Cells:** " + mc_msg)

    # Aktywność populacyjna
    st.subheader("Aktywność populacyjna w czasie")
    BIN_POP = 20.0; n_bins = int(T_MS_POP/BIN_POP)
    t_axis = np.arange(n_bins)*BIN_POP + BIN_POP/2
    fig_p, axes_p = plt.subplots(1,2,figsize=(11,3),sharey=False)
    for k,(gc_i,gc_t) in enumerate(res['gc_spikes']):
        if len(gc_t):
            counts,_ = np.histogram(gc_t, bins=n_bins, range=(0,T_MS_POP))
            axes_p[0].plot(t_axis, counts/(BIN_POP*1e-3*N_GC_r),
                           color=PAT_COLORS[k], lw=1.3, alpha=0.85, label=f'Wzorzec {k+1}')
    axes_p[0].set_title("FR populacyjna — GC", fontsize=9, fontweight='bold')
    axes_p[0].set_xlabel("Czas (ms)"); axes_p[0].set_ylabel("FR (Hz)")
    axes_p[0].legend(fontsize=7); axes_p[0].spines[['top','right']].set_visible(False)

    for k,(fs_i,fs_t) in enumerate(res['fs_spikes']):
        if len(fs_t):
            counts,_ = np.histogram(fs_t, bins=n_bins, range=(0,T_MS_POP))
            axes_p[1].plot(t_axis, counts/(BIN_POP*1e-3*N_FS_r),
                           color=PAT_COLORS[k], lw=1.3, alpha=0.85, label=f'Wzorzec {k+1}')
    axes_p[1].set_title("FR populacyjna — FS", fontsize=9, fontweight='bold')
    axes_p[1].set_xlabel("Czas (ms)"); axes_p[1].set_ylabel("FR (Hz)")
    axes_p[1].legend(fontsize=7); axes_p[1].spines[['top','right']].set_visible(False)
    fig_p.tight_layout(); st.pyplot(fig_p, use_container_width=True); plt.close(fig_p)

    # Dekorelajcja
    st.subheader("Separacja wzorców")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_dec, ax_dec = plt.subplots(figsize=(4,3))
        ax_dec.bar(['R_in\n(wejście)','R_out\n(wyjście GC)'],
                   [res['r_in'], max(res['r_out'],0)],
                   color=['#78909C', COL_GC], edgecolor='black', lw=0.8, width=0.5)
        ax_dec.set_ylim(0,1.05)
        ax_dec.axhline(res['r_in'], color='gray', ls='--', lw=0.8, alpha=0.6)
        ax_dec.annotate('', xy=(1,max(res['r_out'],0)), xytext=(1,res['r_in']),
                        arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.5))
        ax_dec.text(1.32,(res['r_in']+max(res['r_out'],0))/2,
                    f"dec={res['dec']:+.3f}", color='crimson', fontsize=9,
                    va='center', fontweight='bold')
        ax_dec.set_ylabel("Pearson R")
        ax_dec.set_title("Dekorelajcja", fontsize=9, fontweight='bold')
        ax_dec.spines[['top','right']].set_visible(False)
        fig_dec.tight_layout(); st.pyplot(fig_dec, use_container_width=True); plt.close(fig_dec)

    with col_b:
        bin_sizes = [10,25,50,100,250]
        decs = []
        for bms in bin_sizes:
            binned = [bin_spikes(i,t,N_GC_r,float(bms)) for i,t in res['gc_spikes']]
            decs.append(res['r_in'] - mean_pairwise_r(binned))
        fig_ts, ax_ts = plt.subplots(figsize=(4,3))
        ax_ts.plot(bin_sizes, decs, 'o-', color=COL_GC, lw=2, ms=6)
        ax_ts.axhline(0, color='gray', lw=0.6, ls=':')
        ax_ts.set_xscale('log'); ax_ts.set_xticks(bin_sizes)
        ax_ts.set_xticklabels([str(b) for b in bin_sizes], fontsize=8)
        ax_ts.set_xlabel("Bin size (ms)"); ax_ts.set_ylabel("Dekorelajcja")
        ax_ts.set_title("Dekorelajcja vs skala czasowa", fontsize=9, fontweight='bold')
        ax_ts.spines[['top','right']].set_visible(False)
        fig_ts.tight_layout(); st.pyplot(fig_ts, use_container_width=True); plt.close(fig_ts)
