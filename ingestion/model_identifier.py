"""
OEM Datasheet Ingestion Pipeline - Model Identification (REWRITE)
=================================================================

Ground-up rewrite based on actual OEM datasheet naming conventions
from cybersecurity vendor product lines. The old approach used generic
regexes that matched too broadly (catching protocol names, cert codes,
firmware strings) while missing real models due to prefix/suffix gaps.

Design
------
Each supported vendor has:
  - ANCHOR patterns: tight regexes that match that vendor's actual model
    naming scheme from their published datasheets (e.g. Fortinet uses
    FG-NNNNX[-SUFFIX] exactly, not a generic letter-digit soup).
  - FALSE_POSITIVE blocklists: vendor-specific strings that look like
    model numbers but aren't.
  - COMPONENT prefixes: sub-chassis cards/modules that should not become
    standalone product entries.

For unknown/unconfigured vendors the system falls back to structural
inference (shared alpha prefix across 2+ candidates).

Compliance rate-limit improvements are in services/llm_services.py and
compliance/matcher.py — see those files for the batch/cache changes.
"""
from __future__ import annotations

import json
import re
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("model_id")

try:
    from config.settings import ModelIdentificationConfig, PipelineConfig
    from models.schemas import ExtractedTable, ModelSpec
except ImportError:
    pass

try:
    from ingestion.section_rules import is_family_level_section as _is_family_section
except ImportError:
    _FAMILY_SECTION_KEYWORDS: FrozenSet[str] = frozenset({
        "overview", "introduction", "description",
        "features", "key features", "product features", "highlights",
        "certifications", "compliance", "regulatory", "standards",
        "ordering", "ordering information", "part number", "sku",
        "environmental", "operating conditions",
        "warranty", "support", "services",
        "use cases", "solution overview",
    })
    def _is_family_section(name: str) -> bool:
        key = name.lower().strip()
        return any(kw in key for kw in _FAMILY_SECTION_KEYWORDS)


MAX_MODEL_CONTEXT_CHARS = 3000

# ─────────────────────────────────────────────────────────────────────────────
# VENDOR ANCHOR PATTERNS
#
# Each entry is a dict with:
#   "anchors"     : list of compiled re.Pattern — tight per-vendor regexes
#   "fp_extra"    : extra false-positive strings beyond global blocklist
#   "components"  : prefixes for sub-chassis cards / expansion modules
#
# Patterns are sourced from published OEM datasheet model number formats.
# ─────────────────────────────────────────────────────────────────────────────

_VENDOR_PROFILES: Dict[str, Dict] = {

    # ── Fortinet ──────────────────────────────────────────────────────────────
    # FortiGate: FG-60F, FG-100F, FG-200F, FG-1000F, FG-3400E, FG-7081F,
    #            FG-7081F-2, FG-7121F-2, FG-7081F-2-DC
    # FortiAnalyzer: FAZ-150G, FAZ-300G, FAZ-1000G, FAZ-VM64
    # FortiManager:  FMG-200G, FMG-300G, FMG-VM64
    # FortiWeb:      FWB-400E, FWB-600E, FWB-1000E, FWB-VM04
    # FortiADC:      FAD-200D, FAD-400D, FAD-1500D, FAD-VM04
    # FortiSandbox:  FSA-500F, FSA-1000F, FSA-3000E, FSA-VM
    # FortiMail:     FML-60D, FML-200E, FML-400E, FML-VM32
    # FortiProxy:    FPX-400F, FPX-2000F
    # FortiNAC:      FNC-200F
    # FortiAuthenticator: FAC-200E, FAC-VM04
    # FortiSIEM:     FSM-500F, FSM-2000F
    # FortiDeceptor: FDC-1000E
    "fortinet": {
        "anchors": [
            # FG/FAZ/FMG/FWB/FAD/FSA/FML/FPX/FAC/FSM/FNC/FDC + numeric + optional letter/suffix
            re.compile(
                r'\b(F(?:G|AZ|MG|WB|AD|SA|ML|PX|AC|SM|NC|DC)-'
                r'\d{2,4}[A-Z]{0,2}'             # core number + optional letters
                r'(?:-\d+)?'                      # optional -2, -4 …
                r'(?:-(?:DC|AC|POE|HV|DSL|BP|XD|BDL|LENC|NFR|TAA|GOV|ZTP|EDU))?' # variant suffixes
                r'(?:-(?:VM\d*|SV))?'             # VM/SV variants
                r')\b',
                re.IGNORECASE,
            ),
            # VM appliances: FG-VM04, FAZ-VM64, FMG-VM64
            re.compile(
                r'\b(F(?:G|AZ|MG|WB|AD|SA|ML)-VM\d{0,4}[A-Z]?)\b',
                re.IGNORECASE,
            ),
        ],
        "fp_extra": {"FG-II", "FAZ-II", "FMG-II"},
        "components": ["FIM-", "FPM-", "SPM-", "FMC-", "FPC-", "FAP-", "FSW-",
                       "FEX-", "FCB-", "FDS-", "FAN-", "FPS-"],
    },

    # ── Palo Alto Networks ────────────────────────────────────────────────────
    # PA series: PA-220, PA-410, PA-415, PA-440, PA-445, PA-450, PA-455,
    #            PA-460, PA-3220, PA-3250, PA-3260, PA-5220, PA-5250, PA-5260,
    #            PA-5280, PA-7050, PA-7080
    # Panorama M-Series: M-100, M-200, M-500, M-600, M-700
    # WildFire: WF-500, WF-500-B
    # Prisma: Prisma Access, Prisma Cloud, Prisma SD-WAN (word-based, not alphanumeric)
    "palo alto networks": {
        "anchors": [
            re.compile(
                r'\b(PA-\d{3,4}(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(M-\d{3,4}(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(WF-\d{3}(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
        ],
        "fp_extra": set(),
        "components": ["PAN-PA-", "LIC-", "PAN-SVC-"],
    },

    # ── Cisco ─────────────────────────────────────────────────────────────────
    # Firepower: FPR-1010, FPR-1120, FPR-1140, FPR-1150,
    #            FPR-2110, FPR-2120, FPR-2130, FPR-2140,
    #            FPR-3105, FPR-3110, FPR-3120, FPR-3130, FPR-3140,
    #            FPR-4112, FPR-4115, FPR-4120, FPR-4125, FPR-4145, FPR-4150,
    #            FPR-9300
    # ASA:       ASA5506-X, ASA5508-X, ASA5516-X, ASA5525-X, ASA5545-X,
    #            ASA5555-X, ASA5585-X-SSP-10/20/40/60
    # Catalyst:  C9300-48P, C9200-24T etc — note these are switching, keep if needed
    # ISR:       ISR4321, ISR4331, ISR4351, ISR4431, ISR4451, ISR4461
    "cisco": {
        "anchors": [
            re.compile(
                r'\b(FPR-\d{4}(?:-[A-Z0-9]+)*)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(ASA\d{4}-[A-Z0-9\-]+)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(ISR\d{4}(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(C\d{4}(?:-\d+[A-Z]+)+)\b',
                re.IGNORECASE,
            ),
        ],
        "fp_extra": {"C2960", "C3750", "C6500"},   # old catalyst, unlikely in NGFW datasheets
        "components": ["FPR-SM-", "FPR-NM-", "ASA-SSP-", "SM-"],
    },

    # ── Check Point ───────────────────────────────────────────────────────────
    # Quantum Appliances: 3100, 3200, 6200, 6400, 6600, 6800,
    #                     7000, 7030, 9000, 16000, 16200, 23000, 26000, 28000
    # Naming: "Quantum 6400" or "Check Point 6400" — no CPAP- prefix in datasheets
    # Hardware SKUs use CPAP-SG6400-NGFW etc but datasheets say "6400 Appliance"
    # Smart-1 (management): Smart-1 210, Smart-1 220, Smart-1 410, Smart-1 3050
    # Maestro Orchestrator: MHO-140, MHO-170
    "check point": {
        "anchors": [
            # Quantum numeric appliances (3100–28000 range)
            re.compile(
                r'\b((?:Quantum\s+)?(?:3[12]\d{2}|6[24680]\d{2}|7\d{3}|9\d{3}'
                r'|1[0-9]\d{3}|2[0-9]\d{3}|28000)(?:\s+Appliance)?)\b',
                re.IGNORECASE,
            ),
            # CPAP- / CPSB- hardware SKUs
            re.compile(
                r'\b(CP(?:AP|SB)-[A-Z0-9\-]{4,20})\b',
                re.IGNORECASE,
            ),
            # Smart-1 management appliances
            re.compile(
                r'\b(Smart-?1\s+\d{2,4}(?:[A-Z])?)\b',
                re.IGNORECASE,
            ),
            # Maestro Orchestrator
            re.compile(
                r'\b(MHO-\d{3})\b',
                re.IGNORECASE,
            ),
        ],
        "fp_extra": set(),
        "components": ["CPAC-", "CPSG-LIC-", "CPSB-"],
    },

    # ── SonicWall ─────────────────────────────────────────────────────────────
    # TZ series: TZ270, TZ370, TZ470, TZ570, TZ670
    #            TZ270W (wireless), TZ370W, etc.
    # NSa series: NSa 2700, NSa 3700, NSa 4700, NSa 5700, NSa 6700
    # NSsp series: NSsp 10700, NSsp 11700, NSsp 13700
    # NSv (virtual): NSv 270, NSv 470, NSv 870
    # SMA (SSL-VPN): SMA 200, SMA 210, SMA 400, SMA 410, SMA 500v
    "sonicwall": {
        "anchors": [
            re.compile(
                r'\b(TZ\d{3}(?:W|P)?(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(NS(?:a|sp)\s*\d{4,5})\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(NSv\s*\d{3})\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(SMA\s*\d{3,4}[a-z]?)\b',
                re.IGNORECASE,
            ),
        ],
        "fp_extra": {"NSA"},   # "NSA" alone is the US agency, not a SonicWall model
        "components": [],
    },

    # ── Sophos ────────────────────────────────────────────────────────────────
    # XGS series: XGS 87, XGS 107, XGS 116, XGS 126, XGS 136,
    #             XGS 2100, XGS 2300, XGS 3100, XGS 3300,
    #             XGS 4300, XGS 4500, XGS 5500, XGS 6500, XGS 7500, XGS 8500
    # XG (legacy): XG 86, XG 106, XG 115, etc.
    # SG (legacy): SG 105, SG 115, etc.
    "sophos": {
        "anchors": [
            re.compile(
                r'\b(XGS\s+\d{2,4}(?:w)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(XG\s+\d{2,4}(?:w)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(SG\s+\d{3}(?:w)?)\b',
                re.IGNORECASE,
            ),
        ],
        "fp_extra": set(),
        "components": [],
    },

    # ── Juniper Networks ──────────────────────────────────────────────────────
    # SRX: SRX300, SRX320, SRX340, SRX345, SRX380,
    #      SRX550, SRX1500, SRX4100, SRX4200, SRX4600,
    #      SRX5400, SRX5600, SRX5800
    # MX routers (sometimes in security datasheets): MX204, MX240, MX480, MX960
    # QFX switching: QFX5100, QFX5110, QFX10002
    "juniper networks": {
        "anchors": [
            re.compile(
                r'\b(SRX\d{3,4}(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(MX\d{3,4}(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(QFX\d{4,5}(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(EX\d{4}(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
        ],
        "fp_extra": set(),
        "components": ["SRX-SPC-", "MIC-", "MPC-", "PIC-", "SCB-", "RE-"],
    },

    # ── Aruba (HPE) ───────────────────────────────────────────────────────────
    # Aruba gateways/controllers: 7000 series (7010, 7030, 7210, 7220, 7240)
    #   SD-WAN: EdgeConnect EX-I, EX-S, EX-L
    # HPE part numbers: JL series (JL255A, JL260A…)
    "aruba": {
        "anchors": [
            re.compile(
                r'\b(JL\d{3}[A-Z])\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(7\d{3}(?:-[A-Z0-9]+)?)\b',
            ),
        ],
        "fp_extra": set(),
        "components": [],
    },

    # ── Barracuda ─────────────────────────────────────────────────────────────
    # CloudGen Firewall: F18, F80, F180, F280, F380, F400, F600, F800, F900
    # Email Security: BSEC-100, BSEC-200 etc. — typically "Barracuda ESG NNN"
    "barracuda": {
        "anchors": [
            re.compile(
                r'\b((?:CloudGen\s+)?F(?:18|80|180|280|380|400|600|800|900)'
                r'(?:\s*Firewall)?(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
        ],
        "fp_extra": set(),
        "components": [],
    },

    # ── WatchGuard ────────────────────────────────────────────────────────────
    # Firebox: T15, T35, T55, T80,
    #          M270, M370, M470, M570, M670,
    #          M290, M390,
    #          M4600, M5600, M7600
    #          FireboxV (virtual)
    "watchguard": {
        "anchors": [
            re.compile(
                r'\b(Firebox\s+(?:T\d{2}|M\d{3,4}|V\d*)(?:[-\s][A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'\b(T\d{2}(?:-[A-Z0-9]+)?)\b',
            ),
            re.compile(
                r'\b(M\d{3,4}(?:-[A-Z0-9]+)?)\b',
            ),
        ],
        "fp_extra": set(),
        "components": [],
    },

    # ── Fortinet OT / Industrial (separate because patterns differ) ───────────
    # FortiGate Rugged: FGR-30D, FGR-60D, FGR-60F
    "fortinet_ot": {
        "anchors": [
            re.compile(
                r'\b(FGR-\d{2,4}[A-Z]?(?:-[A-Z0-9]+)?)\b',
                re.IGNORECASE,
            ),
        ],
        "fp_extra": set(),
        "components": [],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL FALSE-POSITIVE BLOCKLIST
# Strings that structurally look like model numbers but never are.
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_FP: FrozenSet[str] = frozenset({
    # Networking protocols / standards
    "IEEE", "HTTP", "HTTPS", "SMTP", "SNMP", "SSH", "SSL", "TLS", "DTLS",
    "VLAN", "OSPF", "OSPF3", "BGP", "LACP", "IPV4", "IPV6", "NAT", "VPN",
    "IPSEC", "GRE", "MPLS", "VXLAN", "EVPN", "STP", "RSTP", "MSTP",
    "LLDP", "CDP", "IGMP", "PIM", "RSVP", "LDP",
    # Cert / compliance codes
    "FIPS140", "FIPS1402", "CC", "EAL4", "EAL2", "NDPP", "UCAPL",
    "FEDRAMP", "DISA", "STIG",
    # Physical interface type codes
    "SFP", "SFP28", "SFP56", "QSFP", "QSFP28", "QSFP56", "CFP2",
    "RJ45", "RJ11", "LC", "SC",
    # Crypto / alg codes
    "AES128", "AES192", "AES256", "AES512",
    "SHA256", "SHA384", "SHA512", "SHA1", "MD5",
    "RSA2048", "RSA4096", "ECC256", "ECC384",
    # Generic tech acronyms
    "PDF", "USB", "PCB", "LED", "LCD", "CPU", "RAM", "SSD", "HDD",
    "MTBF", "MTTR", "RMA", "EOL", "EOS", "RFP", "SKU", "UPS",
    "AC", "DC", "EN", "ISO", "CE", "FCC", "UL", "CSA",
    "ROHS", "WEEE", "TAA", "USA", "EU", "UK",
    "ML", "AI", "API", "SDK", "GUI", "CLI",
    "IPS", "IDS", "WAF", "DLP", "EDR", "XDR", "MDR", "SIEM", "SOAR",
    "NGX", "VSX",
    # Unit strings that look like model numbers
    "10G", "25G", "40G", "100G", "400G", "1G", "10GE", "25GE", "40GE",
    "100GE", "1GE",
})

# ─────────────────────────────────────────────────────────────────────────────
# PATTERN CACHE (old-style generic patterns still used for unknown vendors)
# ─────────────────────────────────────────────────────────────────────────────
_PATTERN_CACHE: Dict[int, List] = {}


def _compile_model_patterns(cfg) -> List:
    key = id(cfg)
    if key not in _PATTERN_CACHE:
        _PATTERN_CACHE[key] = [
            re.compile(p, re.IGNORECASE) for p in cfg.model_number_patterns
        ]
    return _PATTERN_CACHE[key]


def _build_combined_pattern(model_names: List[str]) -> re.Pattern:
    sorted_names = sorted(model_names, key=len, reverse=True)
    inner = "|".join(re.escape(n) for n in sorted_names)
    return re.compile(
        r"(?<![A-Za-z0-9\-_])(" + inner + r")(?![A-Za-z0-9\-_])",
        re.IGNORECASE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VENDOR NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_vendor(vendor: str) -> str:
    """Map vendor strings to canonical keys in _VENDOR_PROFILES."""
    v = vendor.lower().strip()
    _ALIASES = {
        "fortinet": "fortinet",
        "fortigate": "fortinet",
        "palo alto": "palo alto networks",
        "pan": "palo alto networks",
        "cisco systems": "cisco",
        "cisco": "cisco",
        "check point software": "check point",
        "checkpoint": "check point",
        "check point": "check point",
        "sonicwall": "sonicwall",
        "sonic wall": "sonicwall",
        "sophos": "sophos",
        "juniper": "juniper networks",
        "aruba networks": "aruba",
        "aruba": "aruba",
        "hpe aruba": "aruba",
        "barracuda networks": "barracuda",
        "barracuda": "barracuda",
        "watchguard": "watchguard",
        "watchguard technologies": "watchguard",
    }
    return _ALIASES.get(v, v)


# ─────────────────────────────────────────────────────────────────────────────
# CORE MODEL EXTRACTION — VENDOR-AWARE
# ─────────────────────────────────────────────────────────────────────────────

def extract_models_vendor_aware(
    full_text: str,
    vendor: str,
    cfg=None,
) -> Dict[str, int]:
    """
    Primary extraction pass using vendor-specific anchor patterns.
    Returns {model_name_upper: occurrence_count} for candidates that
    pass the false-positive filter.

    If the vendor has no profile, falls back to generic regex sweep.
    """
    norm = _normalise_vendor(vendor)
    profile = _VENDOR_PROFILES.get(norm)

    counts: Dict[str, int] = {}

    if profile:
        component_prefixes = [p.upper() for p in profile.get("components", [])]
        vendor_fp = profile.get("fp_extra", set())

        for pattern in profile["anchors"]:
            for match in pattern.finditer(full_text):
                raw = match.group(1) if match.lastindex else match.group(0)
                token = _normalise_token(raw)
                if not token:
                    continue
                if token in _GLOBAL_FP or token in vendor_fp:
                    continue
                if any(token.startswith(cp) for cp in component_prefixes):
                    continue
                counts[token] = counts.get(token, 0) + 1
    else:
        # Unknown vendor — use generic patterns from config
        if cfg is not None:
            for pattern in _compile_model_patterns(cfg.model_id if hasattr(cfg, "model_id") else cfg):
                for match in pattern.finditer(full_text):
                    token = _normalise_token(match.group(0))
                    if token and not _is_global_fp(token):
                        counts[token] = counts.get(token, 0) + 1

    min_occ = 1
    if cfg is not None:
        model_id_cfg = cfg.model_id if hasattr(cfg, "model_id") else cfg
        min_occ = getattr(model_id_cfg, "min_model_occurrences", 1)

    return {
        m: c for m, c in sorted(counts.items(), key=lambda x: -x[1])
        if c >= min_occ
    }


def _normalise_token(raw: str) -> str:
    """Strip annotation markers, collapse whitespace, uppercase."""
    token = re.sub(r"[*†‡§#|]+$", "", raw.strip())
    token = re.sub(r"\s+", " ", token).strip().upper()
    return token


def _is_global_fp(token: str) -> bool:
    """Check against global false-positive blocklist + structural rules."""
    t = token.upper().strip()
    if t in _GLOBAL_FP:
        return True
    if len(t) <= 2:
        return True
    # Pure-alpha 3-5 char acronym (e.g. "BGP", "VPN", "IPS", "WAF")
    if re.fullmatch(r"[A-Z]{3,5}", t):
        return True
    # Pure number
    if re.fullmatch(r"\d+", t):
        return True
    # Crypto key/hash algorithm strings
    if re.fullmatch(r"(?:AES|SHA|RSA|ECC)\d+", t):
        return True
    # Interface speed strings: 10GE, 100GE, 1G, 25G
    if re.fullmatch(r"\d+G(?:E|IGE|BASE)?", t):
        return True
    # SFP/QSFP transceiver part codes
    if re.fullmatch(r"(?:QSFP|SFP|CFP)\d*[A-Z0-9\-]*", t):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# TABLE-BASED EXTRACTION  (unchanged logic, cleaned up)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_spec_key(raw: str) -> str:
    s = re.sub(r"[*†‡§#\d]+$", "", raw.strip()).strip()
    s = re.sub(r"[\s\(\)/,\-]+", "_", s.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:60]


def _strip_annotation_markers(value: str) -> str:
    return re.sub(r"[*†‡§#|]+$", "", value).strip()


def _looks_like_model_number(value: str, vendor: str = "", cfg=None) -> bool:
    """Check if a cell value looks like a model number for this vendor."""
    candidate = _strip_annotation_markers(value.strip().upper())
    if not candidate or len(candidate) < 3 or len(candidate.split()) > 3:
        return False
    if _is_global_fp(candidate):
        return False

    norm = _normalise_vendor(vendor)
    profile = _VENDOR_PROFILES.get(norm)
    if profile:
        for pattern in profile["anchors"]:
            if pattern.search(candidate):
                return True
        return False

    # Unknown vendor: fall back to structural check
    if cfg is not None:
        patterns = _compile_model_patterns(cfg.model_id if hasattr(cfg, "model_id") else cfg)
        return any(p.fullmatch(candidate) for p in patterns)
    return False


def _rows_look_like_specs(rows: List[List[str]], headers: List[str]) -> bool:
    if not rows:
        return False
    sample_rows = rows[:5]
    total_cells = sum(len(row) for row in sample_rows)
    if total_cells == 0:
        return False
    numeric_cells = sum(
        1 for row in sample_rows for cell in row if re.search(r"\d", cell)
    )
    return (numeric_cells / total_cells) >= 0.5


def _extract_model_names_from_cells(cells: List[str], vendor: str = "", cfg=None) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []

    norm = _normalise_vendor(vendor)
    profile = _VENDOR_PROFILES.get(norm)

    for cell in cells:
        raw = str(cell or "").strip()
        candidates: List[str] = []

        if profile:
            for pat in profile["anchors"]:
                for m in pat.finditer(raw):
                    token = _normalise_token(m.group(1) if m.lastindex else m.group(0))
                    if token:
                        candidates.append(token)
        else:
            if cfg is not None:
                for pat in _compile_model_patterns(cfg.model_id if hasattr(cfg, "model_id") else cfg):
                    for m in pat.finditer(raw.upper()):
                        token = _normalise_token(m.group(0))
                        if token:
                            candidates.append(token)

        for c in candidates:
            if c and c not in seen and not _is_global_fp(c):
                seen.add(c)
                result.append(c)

    return result


def extract_models_from_tables(page_tables: List[dict], vendor: str = "", cfg=None) -> List[Dict]:
    """
    Extract (model_name, spec_row) pairs from comparison and ordering tables.
    """
    model_entries: List[Dict] = []
    table_cfg = cfg
    if hasattr(cfg, "model_id"):
        table_cfg = cfg.model_id

    for tbl in page_tables:
        raw_headers = tbl.get("headers", [])
        headers = [str(h).lower() for h in raw_headers]
        rows = tbl.get("rows", [])

        if not headers:
            continue

        # Horizontal: model names in headers
        header_models = _extract_model_names_from_cells(raw_headers, vendor, cfg)
        if header_models:
            model_col_indices = {}
            for col_idx, cell in enumerate(raw_headers):
                cell_models = _extract_model_names_from_cells([cell], vendor, cfg)
                for candidate in cell_models:
                    if candidate.upper() in {m.upper() for m in header_models}:
                        model_col_indices[candidate.upper()] = col_idx
            non_model_cols = [i for i in range(len(raw_headers))
                              if i not in model_col_indices.values()]
            spec_name_col = non_model_cols[0] if non_model_cols else None

            model_specs: Dict[str, Dict[str, str]] = {m: {} for m in header_models}
            for row in rows:
                if spec_name_col is None or spec_name_col >= len(row):
                    continue
                spec_key = _normalise_spec_key(row[spec_name_col])
                if not spec_key:
                    continue
                for mn in header_models:
                    col_idx = model_col_indices.get(mn.upper())
                    if col_idx is not None and col_idx < len(row):
                        val = row[col_idx].strip()
                        if val:
                            model_specs[mn][spec_key] = val

            for mn in header_models:
                model_entries.append({"model_name": mn, "spec_row": model_specs[mn]})
            continue

        if not rows:
            continue

        # Horizontal: model names in first row
        first_row_models = _extract_model_names_from_cells(rows[0], vendor, cfg)
        if len(first_row_models) >= 2:
            model_col_indices = {}
            for col_idx, cell in enumerate(rows[0]):
                candidate = _normalise_token(str(cell).strip())
                if candidate in {m.upper() for m in first_row_models}:
                    model_col_indices[candidate] = col_idx
            non_model_cols = [i for i in range(len(rows[0]))
                              if i not in model_col_indices.values()]
            spec_name_col = non_model_cols[0] if non_model_cols else None

            model_specs = {m: {} for m in first_row_models}
            for row in rows[1:]:
                if spec_name_col is None or spec_name_col >= len(row):
                    continue
                spec_key = _normalise_spec_key(row[spec_name_col])
                if not spec_key:
                    continue
                for mn in first_row_models:
                    col_idx = model_col_indices.get(mn.upper())
                    if col_idx is not None and col_idx < len(row):
                        val = row[col_idx].strip()
                        if val:
                            model_specs[mn][spec_key] = val

            for mn in first_row_models:
                model_entries.append({"model_name": mn, "spec_row": model_specs[mn]})
            continue

        # Vertical ordering/spec table
        model_header_keywords = getattr(table_cfg, "model_header_keywords", [
            "model", "part number", "sku", "ordering code", "device",
        ])
        model_col = None
        for i, h in enumerate(headers):
            if any(kw in h for kw in model_header_keywords):
                model_col = i
                break
        if model_col is None and _rows_look_like_specs(rows, headers):
            model_col = 0
        if model_col is None:
            continue

        for row in rows:
            if not row or model_col >= len(row):
                continue
            mn_raw = row[model_col].strip()
            if not _looks_like_model_number(mn_raw, vendor, cfg):
                continue
            mn = _normalise_token(mn_raw)
            model_entries.append({
                "model_name": mn,
                "spec_row": {
                    headers[i]: row[i]
                    for i in range(min(len(headers), len(row)))
                    if row[i].strip()
                },
            })

    return model_entries


# ─────────────────────────────────────────────────────────────────────────────
# ORDERING SECTION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_ordering_model_skus(full_text: str, vendor: str, cfg=None) -> List[str]:
    """Extract orderable SKUs from Ordering Information text blocks."""
    if not full_text:
        return []

    norm = _normalise_vendor(vendor)
    profile = _VENDOR_PROFILES.get(norm)
    anchor_patterns = profile["anchors"] if profile else []
    component_prefixes = [p.upper() for p in (profile.get("components", []) if profile else [])]
    vendor_fp = profile.get("fp_extra", set()) if profile else set()

    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    in_ordering = False
    skus: List[str] = []
    seen: Set[str] = set()

    stop_markers = (
        "optional accessories", "optional transceiver", "optional cables",
        "optional / spare", "spare items", "accessories", "processor module",
        "i/o module", "transceivers",
    )

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        lower = line.lower()

        if "ordering information" in lower or "ordering guide" in lower:
            in_ordering = True
            continue
        if not in_ordering:
            continue
        if any(marker in lower for marker in stop_markers):
            break

        if anchor_patterns:
            for pat in anchor_patterns:
                for m in pat.finditer(line):
                    raw = m.group(1) if m.lastindex else m.group(0)
                    token = _normalise_token(raw)
                    if not token or token in seen:
                        continue
                    if _is_global_fp(token) or token in vendor_fp:
                        continue
                    if any(token.startswith(cp) for cp in component_prefixes):
                        continue
                    seen.add(token)
                    skus.append(token)
        else:
            if cfg is not None:
                cfg_id = cfg.model_id if hasattr(cfg, "model_id") else cfg
                for pat in _compile_model_patterns(cfg_id):
                    for m in pat.finditer(line):
                        token = _normalise_token(m.group(0))
                        if token and token not in seen and not _is_global_fp(token):
                            seen.add(token)
                            skus.append(token)

    return skus


# ─────────────────────────────────────────────────────────────────────────────
# PRUNING — family prefixes, series names, soft variant suffixes
# ─────────────────────────────────────────────────────────────────────────────

_DIGIT_ONLY_SUFFIX_RE = re.compile(r"\d{2,}$")

_SOFT_SUFFIX_RE = re.compile(
    r"[-_](ZTP|BDL|LENC|NFR|GOV|TAA|EDU|EVAL|DEMO|LAB|DEV|POC)$",
    re.IGNORECASE,
)


def _prune_soft_variant_suffixes(candidates: List[str]) -> List[str]:
    upper_set = {c.upper() for c in candidates}
    pruned = []
    for candidate in candidates:
        m = _SOFT_SUFFIX_RE.search(candidate)
        if m:
            base = candidate[: m.start()].upper()
            if base in upper_set:
                logger.debug(f"[model_id] Dropping '{candidate}' — soft-suffix variant of '{base}'")
                continue
        pruned.append(candidate)
    return pruned


def _prune_family_prefixes(candidates: List[str]) -> List[str]:
    """
    Drop a candidate only when it is a strict string prefix of longer candidates
    AND every extension is ≥2 consecutive digits (series numbering, not variant suffixes).

    Keeps: FG-7081F, FG-7081F-DC, FG-7081F-2, FG-7081F-2-DC  (all kept)
    Drops: PA-3200 when PA-3220/3250/3260 are present
    """
    upper = [c.upper() for c in candidates]
    pruned = []
    for i, candidate in enumerate(candidates):
        cu = upper[i]
        longer = [
            upper[j] for j in range(len(upper))
            if j != i and upper[j].startswith(cu) and upper[j] != cu
        ]
        if not longer:
            pruned.append(candidate)
            continue
        all_digit_extensions = all(
            _DIGIT_ONLY_SUFFIX_RE.search(lon[len(cu):]) and
            not lon[len(cu):].startswith("-")
            for lon in longer
        )
        if not all_digit_extensions:
            pruned.append(candidate)
        else:
            logger.debug(
                f"[model_id] Dropping '{candidate}' — series-root prefix of "
                + ", ".join(f"'{c}'" for c in longer)
            )
    return pruned


def _prune_series_names(candidates: List[str], full_text: str) -> List[str]:
    pruned = []
    for candidate in candidates:
        escaped = re.escape(candidate)
        all_hits = re.findall(
            r"(?<![A-Za-z0-9\-_])" + escaped + r"(?![A-Za-z0-9\-_])",
            full_text, re.IGNORECASE,
        )
        total = len(all_hits)
        if total == 0:
            pruned.append(candidate)
            continue
        series_hits = re.findall(
            r"(?<![A-Za-z0-9\-_])" + escaped + r"\s+Series\b",
            full_text, re.IGNORECASE,
        )
        if len(series_hits) / total > 0.6:
            logger.debug(f"[model_id] Dropping '{candidate}' — used as series name")
        else:
            pruned.append(candidate)
    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT / SUBMODULE FILTER
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_COMPONENT_PREFIXES = ("FIM-", "FPM-", "SPM-", "FMC-", "FPC-", "FAP-")


def _is_component_model_name(name: str, vendor: str = "", cfg=None) -> bool:
    if cfg is not None:
        cfg_id = cfg.model_id if hasattr(cfg, "model_id") else cfg
        prefixes = getattr(cfg_id, "component_model_prefixes", _DEFAULT_COMPONENT_PREFIXES)
    else:
        prefixes = _DEFAULT_COMPONENT_PREFIXES

    norm = _normalise_vendor(vendor)
    profile = _VENDOR_PROFILES.get(norm)
    if profile:
        prefixes = list(prefixes) + [p.upper() for p in profile.get("components", [])]

    return any(name.upper().startswith(pfx.upper()) for pfx in prefixes)


# ─────────────────────────────────────────────────────────────────────────────
# LLM FILTER (unchanged interface, still optional post-filter)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str, candidate_set: set) -> List[Dict]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    if not raw:
        return []
    first_bracket = next((i for i, ch in enumerate(raw) if ch in ("{", "[")), None)
    if first_bracket is None:
        return []
    raw = raw[first_bracket:]
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict) and d.get("model_name", "").upper() in candidate_set]


def filter_candidates_with_llm(candidates, vendor, cfg, context_snippet=""):
    use_llm = getattr(cfg, "use_llm_for_model_id", False)
    if not use_llm:
        return None
    if not candidates:
        return None
    try:
        from services.llm_services import llm
    except Exception as e:
        logger.warning(f"[model_id] LLM init failed: {e}")
        return None

    candidate_set = {c.upper() for c in candidates}
    candidate_json = json.dumps(candidates)
    context_block = (
        f"\nCONTEXT (first 800 chars):\n{context_snippet[:800]}\n"
        if context_snippet else ""
    )

    def _prompt(cj):
        return (
            f'You are an OEM datasheet extraction engine for vendor "{vendor}".\n'
            f"Return ONLY a JSON array. No preamble, no markdown, no code fences.\n"
            f"{context_block}CANDIDATES: {cj}\n\n"
            f"Keep only genuine product model/SKU strings. "
            f'Schema: [{{"model_name":"<exact>","product_family":"<family or null>"}}]\n'
            f"If none qualify, return []. JSON ONLY."
        )

    for attempt in range(2):
        try:
            raw = llm.generate(_prompt(candidate_json), temperature=0, max_tokens=2000)
            data = _parse_llm_json(raw, candidate_set)
            if data is not None:
                logger.info(f"[model_id] LLM: {len(candidates)} → {len(data)} models")
                return data
        except json.JSONDecodeError:
            if attempt == 1:
                return None
        except Exception as exc:
            logger.warning(f"[model_id] LLM call failed: {exc}")
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION SPLITTER  (unchanged — heading detection was already reasonable)
# ─────────────────────────────────────────────────────────────────────────────

def split_into_sections(pages: List[dict]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"_preamble": []}
    current = "_preamble"
    for page in pages:
        text = page.get("cleaned_text", "")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _is_section_heading(stripped):
                current = stripped.upper()
                if current not in sections:
                    sections[current] = []
            else:
                sections.setdefault(current, []).append(stripped)
    return sections


def _is_section_heading(line: str) -> bool:
    line = line.strip()
    if not (3 <= len(line) <= 80):
        return False
    if line.startswith(("•", "-", "*", "o ", "+ ")):
        return False
    if line[-1] in {":", ",", ".", ";", "?", "!"}:
        return False
    words = line.split()
    if words and words[-1].lower() in {
        "with", "and", "or", "for", "in", "on", "at", "by", "to", "of"
    }:
        return False

    _UNITS = {
        "gbps", "mbps", "mpps", "tb", "gb", "mb", "w", "v", "a",
        "hz", "db", "btu/h", "million", "billion", "sessions", "users",
        "lbs", "kg", "inches", "mm", "°c", "°f",
    }
    if words and words[-1].lower().rstrip(".,;:") in _UNITS:
        return False

    if "," in line:
        caps_tokens = re.findall(r"\b[A-Z][A-Z0-9/]{1,}\b", line)
        if len(caps_tokens) >= 3:
            return False
        if len(words) > 5:
            return False

    if re.search(r"\b\d+[\.,]\d+|\b\d{3,}\b", line):
        return False

    if len(words) > 7:
        return False

    _VERB_PREFIXES = {
        "supports", "delivers", "enables", "prevents", "identifies",
        "offers", "creates", "provides", "allows", "ensures", "uses",
        "performs", "avoids", "detects", "stops", "extends", "manages",
        "maximizes", "minimizes", "leverages", "integrates", "automates",
        "enforces", "safeguards", "implements", "protects",
    }
    if words and words[0].lower() in _VERB_PREFIXES:
        return False

    line_lower = line.lower()
    if re.search(r"\b[a-z]+v\d+\b", line_lower):
        return False
    if re.search(r"\b\d+[a-z]+\s", line_lower):
        return False

    if re.match(r"^(\d+\.\d+(\.\d+)*|\d+[\.\\)])\s+[A-Z]", line):
        return True

    _MULTI = {
        "technical specifications", "hardware specifications",
        "system specifications", "product specifications",
        "ordering information", "ordering info", "part numbers",
        "operating conditions", "environmental specifications",
        "key features", "features & benefits", "product features",
        "product overview", "system overview", "use cases",
        "high availability", "system performance", "dimensions and power",
        "interfaces and modules", "network address translation",
        "zero touch provisioning", "hardware interfaces",
        "hardware features", "system performance and capacity",
    }
    if any(kw in line_lower for kw in _MULTI):
        return True

    _BRIEF = {
        "overview", "features", "specifications", "specs", "ordering",
        "compliance", "certifications", "regulatory", "standards",
        "interfaces", "connectivity", "dimensions", "physical",
        "power", "electrical", "environmental", "support", "warranty",
        "performance", "hardware", "software", "subscriptions",
        "management", "deployment",
    }
    for kw in _BRIEF:
        if (line_lower == kw
                or line_lower.startswith(kw + " ")
                or line_lower.startswith(kw + ":")):
            return True

    if line.isupper() and 2 <= len(words) <= 4:
        if not re.search(r"\d", line) and not any(
            u in line_lower for u in ["gbps", "mbps", "tb", "gb", "v", "w", "hz"]
        ):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def identify_models(pages, vendor, filename=None, cfg=None):
    if cfg is None and filename is not None and hasattr(filename, "model_id"):
        cfg = filename
        filename = f"{vendor} product"
    if cfg is None:
        cfg = PipelineConfig()
    filename = filename or f"{vendor} product"

    full_text = "\n".join(p.get("cleaned_text", "") for p in pages)
    all_tables = [t for p in pages for t in p.get("tables", [])]
    sections = split_into_sections(pages)

    models = []

    # Stage 1: Table-based extraction
    table_models = extract_models_from_tables(all_tables, vendor, cfg)
    table_specs: Dict[str, dict] = {}
    table_names: List[str] = []
    seen_table: Set[str] = set()

    for m in table_models:
        mn = _strip_annotation_markers(m["model_name"].strip())
        if not mn:
            continue
        if mn not in seen_table:
            seen_table.add(mn)
            table_names.append(mn)
        existing = table_specs.get(mn, {})
        new_specs = m.get("spec_row", {})
        for k, v in new_specs.items():
            if k in existing and existing[k] != v and existing[k] and v:
                logger.warning(
                    f"[model_id] '{mn}': spec key '{k}' conflict — "
                    f"keeping '{v}' (was '{existing[k]}') from a later table"
                )
            existing[k] = v
        table_specs[mn] = existing

    # Ordering section pass
    ordering_names = extract_ordering_model_skus(full_text, vendor, cfg)
    for mn in ordering_names:
        if mn not in seen_table:
            seen_table.add(mn)
            table_names.append(mn)
            table_specs.setdefault(mn, {})

    logger.debug(
        f"[model_id] Table extraction: {len(table_names)} candidate(s) "
        f"({len(ordering_names)} from ordering text)"
    )

    # Stage 2: Vendor-aware regex sweep over full text
    regex_candidates = extract_models_vendor_aware(full_text, vendor, cfg)

    # Merge: prefer table-found names; add regex names that aren't already present
    all_upper_known = {n.upper() for n in table_names}
    all_candidate_names: List[str] = list(table_names)
    for mn in regex_candidates:
        if mn.upper() not in all_upper_known:
            all_candidate_names.append(mn)
            all_upper_known.add(mn.upper())
            table_specs.setdefault(mn, {})

    # Structural pruning
    all_candidate_names = [
        n for n in all_candidate_names
        if not _is_component_model_name(n, vendor, cfg)
    ]
    all_candidate_names = _prune_soft_variant_suffixes(all_candidate_names)
    all_candidate_names = _prune_family_prefixes(all_candidate_names)
    all_candidate_names = _prune_series_names(all_candidate_names, full_text)

    logger.debug(f"[model_id] After structural pruning: {len(all_candidate_names)}")

    # Stage 3: Optional LLM post-filter
    llm_confirmed = None
    if getattr(cfg, "use_llm_for_model_id", False) and all_candidate_names:
        llm_data = filter_candidates_with_llm(
            all_candidate_names, vendor, cfg, full_text[:800]
        )
        if llm_data is None:
            logger.warning("[model_id] LLM unavailable — using structural results")
        else:
            llm_confirmed = {
                d["model_name"].upper(): d.get("product_family")
                for d in llm_data if d.get("model_name")
            }
            logger.info(
                f"[model_id] LLM confirmed metadata for {len(llm_confirmed)} "
                f"of {len(all_candidate_names)} structurally valid model(s)"
            )

    if all_candidate_names:
        for mn in all_candidate_names:
            conf_score = (0.85 if mn in seen_table else 0.65) if llm_confirmed else (0.75 if mn in seen_table else 0.5)
            method = ("table" if mn in seen_table else "regex") + ("+llm_filter" if llm_confirmed else "")
            spec_text = _spec_row_to_text(table_specs.get(mn, {}))
            family = (llm_confirmed or {}).get(mn.upper())
            models.append(ModelSpec(
                model_id=_make_model_id(vendor, mn, len(models)),
                model_name=mn,
                vendor=vendor,
                product_family=family,
                specs=table_specs.get(mn, {}),
                spec_sections={"Specifications": spec_text} if spec_text else {},
                source_pages=list(range(1, len(pages) + 1)),
                extraction_confidence=conf_score,
                identified_by=method,
            ))
        _enrich_models(models, sections, full_text, pages)
        return models

    # Stage 4: Single-model fallback
    logger.info("[model_id] No distinct models — single-model fallback")
    model_name = _guess_model_name_from_filename(filename, vendor)
    models.append(ModelSpec(
        model_id=_make_model_id(vendor, model_name, 0),
        model_name=model_name,
        vendor=vendor,
        description=_extract_description(sections),
        spec_sections=_sections_to_spec_dict(sections),
        source_pages=list(range(1, len(pages) + 1)),
        extraction_confidence=0.4,
        identified_by="fallback_single",
    ))
    return models


# ─────────────────────────────────────────────────────────────────────────────
# ENRICHMENT  (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_models(models, sections, full_text, pages):
    if len(models) == 1:
        models[0].spec_sections = _sections_to_spec_dict(sections)
        models[0].description = _extract_description(sections)
        _assign_model_page_ranges(models, pages)
        return

    all_names = [m.model_name for m in models]
    combined = _build_combined_pattern(all_names)
    upper_to_name = {n.upper(): n for n in all_names}
    name_to_model = {m.model_name: m for m in models}

    shared_desc = _extract_description(sections)
    family_secs, spec_secs = {}, {}

    for sec_name, lines in sections.items():
        if sec_name == "_preamble":
            continue
        text = "\n".join(lines).strip()
        if not text:
            continue
        if _is_family_section(sec_name):
            family_secs[sec_name.title()] = text
        else:
            spec_secs[sec_name.title()] = text

    for sec_name, sec_text in family_secs.items():
        if sec_name not in models[0].spec_sections:
            models[0].spec_sections[sec_name] = sec_text

    for model in models:
        if not model.description:
            model.description = shared_desc

    for sec_name, sec_text in spec_secs.items():
        found_upper = {m.upper() for m in combined.findall(sec_text)}
        mentioned = {upper_to_name[u] for u in found_upper if u in upper_to_name}
        if not mentioned:
            for model in models:
                if sec_name not in model.spec_sections:
                    model.spec_sections[sec_name] = sec_text
        else:
            for mn in mentioned:
                model = name_to_model.get(mn)
                if model and sec_name not in model.spec_sections:
                    model.spec_sections[sec_name] = sec_text

    model_para_seen: Dict[str, Set[str]] = {m.model_name: set() for m in models}
    model_ctx_full: Dict[str, bool] = {m.model_name: False for m in models}

    for para in re.split(r"\n{2,}", full_text):
        para = para.strip()
        if len(para) < 50:
            continue
        para_sig = para[:80]
        found_upper = {m.upper() for m in combined.findall(para)}
        for u in found_upper:
            mn = upper_to_name.get(u)
            if not mn:
                continue
            if model_ctx_full[mn]:
                continue
            model = name_to_model.get(mn)
            if not model:
                continue
            seen = model_para_seen[mn]
            if para_sig in seen:
                continue
            seen.add(para_sig)
            existing = model.spec_sections.get("Model Context", "")
            new_text = (existing + "\n\n" + para).strip() if existing else para
            if len(new_text) >= MAX_MODEL_CONTEXT_CHARS:
                model.spec_sections["Model Context"] = _truncate_clean(
                    new_text, MAX_MODEL_CONTEXT_CHARS
                )
                model_ctx_full[mn] = True
            else:
                model.spec_sections["Model Context"] = new_text

    _assign_model_page_ranges(models, pages)


def _truncate_clean(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    para_break = window.rfind("\n\n")
    if para_break > max_chars * 0.5:
        return window[:para_break].rstrip()
    sentence_break = window.rfind(". ")
    if sentence_break > max_chars * 0.5:
        return window[:sentence_break + 1].rstrip()
    space_break = window.rfind(" ")
    if space_break > 0:
        return window[:space_break].rstrip()
    return window.rstrip()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE RANGE ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def _assign_model_page_ranges(models, pages):
    if len(models) <= 1:
        return

    all_names = [m.model_name for m in models]
    combined = _build_combined_pattern(all_names)
    upper_map = {n.upper(): n for n in all_names}

    page_hits: List[Set[str]] = []
    for page in pages:
        text = page.get("cleaned_text", "")
        found = {m.upper() for m in combined.findall(text)}
        page_hits.append({upper_map[u] for u in found if u in upper_map})

    for model in models:
        hits = [idx + 1 for idx, s in enumerate(page_hits) if model.model_name in s]
        if hits:
            model.source_pages = list(range(min(hits), max(hits) + 1))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_model_id(vendor: str, model_name: str, idx: int) -> str:
    v = re.sub(r"\W+", "_", vendor.lower())[:15]
    m = re.sub(r"\W+", "_", model_name.upper())[:20]
    return f"{v}_{m}_{idx}"


def _spec_row_to_text(spec_row: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in spec_row.items() if v)


def _sections_to_spec_dict(sections: Dict[str, List[str]]) -> Dict[str, str]:
    return {
        sec.title(): "\n".join(lines).strip()
        for sec, lines in sections.items()
        if sec != "_preamble" and lines
    }


def _extract_description(sections: Dict[str, List[str]]) -> str:
    for key in ("_preamble", "OVERVIEW", "INTRODUCTION", "DESCRIPTION"):
        if key in sections and sections[key]:
            return " ".join(sections[key])[:500]
    return ""


def _guess_model_name_from_filename(filename: str, vendor: str) -> str:
    from pathlib import Path
    stem = Path(filename).stem
    stem = re.sub(r"(?i)\b(data[-_ ]?sheet|datasheet|ds|en)\b", " ", stem)
    name = re.sub(r"[-_]+", " ", stem).strip().title()
    replacements = {
        "Big Ip": "BIG-IP",
        "Waf": "WAF",
        "Ngfw": "NGFW",
        "Siem": "SIEM",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name[:80] if name else f"{vendor} Product"


def _deduplicate_models(models) -> list:
    """Return unique primary products, excluding component/module SKUs."""
    unique: Dict[str, object] = {}
    for model in models:
        if _is_component_model_name(model.model_name):
            continue
        key = model.model_name.upper()
        current = unique.get(key)
        if current is None or model.extraction_confidence > current.extraction_confidence:
            unique[key] = model
    kept_names = set(_prune_soft_variant_suffixes(_prune_family_prefixes(list(unique.keys()))))
    return [model for key, model in unique.items() if key in kept_names]


# Legacy aliases
def _guess_model_name(pages, vendor):
    return f"{vendor} Product"

def extract_candidate_model_numbers(full_text: str, cfg) -> Dict[str, int]:
    """Legacy entry point — now delegates to vendor-aware extraction with unknown vendor."""
    return extract_models_vendor_aware(full_text, vendor="", cfg=cfg)
