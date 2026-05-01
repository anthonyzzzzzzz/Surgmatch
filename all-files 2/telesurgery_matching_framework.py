"""
Telesurgery Matching Framework
==============================
A lightweight matching engine that connects surgical cases
with the best-fit remote surgeon/proctor.

Matching Logic:
  Layer 1 - Hard Filter:  license, device, availability
  Layer 2 - Quality Score: procedure experience, outcomes, subspecialty fit
  Layer 3 - Feasibility:   network latency, cost budget
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =============================================================================
# 1. ENUMS & CONSTANTS
# =============================================================================

class Specialty(Enum):
    OPHTHALMOLOGY = "Ophthalmology"
    UROLOGY = "Urology"
    GENERAL_SURGERY = "General Surgery"
    GYNECOLOGY = "Gynecology"
    ORTHOPEDICS = "Orthopedics"


class DeviceType(Enum):
    DAVINCI_XI = "da Vinci Xi"
    DAVINCI_SP = "da Vinci SP"
    HUGO_RAS = "Medtronic Hugo RAS"
    VERSIUS = "CMR Versius"
    TOUMAI = "Toumai"


class ComplexityLevel(Enum):
    LOW = 1       # routine / straightforward
    MEDIUM = 2    # moderate difficulty
    HIGH = 3      # complex, comorbidities, revision
    CRITICAL = 4  # rare / high-risk


MAX_SAFE_LATENCY_MS = 150  # above this, telesurgery safety degrades


# =============================================================================
# 2. PROCEDURE TAXONOMY
#    Anchored to CPT codes where applicable
# =============================================================================

@dataclass
class Procedure:
    code: str               # CPT code or internal ID
    name: str
    specialty: Specialty
    subspecialty: str        # e.g. "Glaucoma", "Retina", "Cataract"
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        return self.code == other.code


# --- Ophthalmology Example Taxonomy ---

PROCEDURES = {
    # Cataract
    "66984": Procedure("66984", "Cataract removal with IOL (standard)", Specialty.OPHTHALMOLOGY, "Cataract", ComplexityLevel.LOW),
    "66982": Procedure("66982", "Cataract removal with IOL (complex)", Specialty.OPHTHALMOLOGY, "Cataract", ComplexityLevel.HIGH),
    "66987": Procedure("66987", "Cataract removal combined with other procedure", Specialty.OPHTHALMOLOGY, "Cataract", ComplexityLevel.HIGH),

    # Glaucoma
    "66711": Procedure("66711", "MIGS - trabecular bypass", Specialty.OPHTHALMOLOGY, "Glaucoma", ComplexityLevel.MEDIUM),
    "66170": Procedure("66170", "Trabeculectomy", Specialty.OPHTHALMOLOGY, "Glaucoma", ComplexityLevel.HIGH),
    "66183": Procedure("66183", "Glaucoma drainage device (aqueous shunt)", Specialty.OPHTHALMOLOGY, "Glaucoma", ComplexityLevel.HIGH),
    "65855": Procedure("65855", "Laser trabeculoplasty (SLT/ALT)", Specialty.OPHTHALMOLOGY, "Glaucoma", ComplexityLevel.LOW),

    # Retina
    "67036": Procedure("67036", "Pars plana vitrectomy (PPV)", Specialty.OPHTHALMOLOGY, "Retina", ComplexityLevel.HIGH),
    "67108": Procedure("67108", "Retinal detachment repair (scleral buckle + PPV)", Specialty.OPHTHALMOLOGY, "Retina", ComplexityLevel.CRITICAL),
    "67228": Procedure("67228", "Laser photocoagulation of retina", Specialty.OPHTHALMOLOGY, "Retina", ComplexityLevel.MEDIUM),

    # Cornea
    "65710": Procedure("65710", "Corneal transplant (lamellar)", Specialty.OPHTHALMOLOGY, "Cornea", ComplexityLevel.HIGH),
    "0542T": Procedure("0542T", "Corneal cross-linking", Specialty.OPHTHALMOLOGY, "Cornea", ComplexityLevel.MEDIUM),

    # Oculoplastics
    "67901": Procedure("67901", "Blepharoptosis repair", Specialty.OPHTHALMOLOGY, "Oculoplastics", ComplexityLevel.MEDIUM),
}


# =============================================================================
# 3. DATA MODELS
# =============================================================================

@dataclass
class SurgeonProfile:
    id: str
    name: str
    specialty: Specialty
    subspecialties: list[str]                     # e.g. ["Glaucoma", "Cataract"]
    licensed_states: list[str]                     # e.g. ["CA", "NY", "TX"]
    certified_devices: list[DeviceType]
    procedure_log: dict[str, int]                  # CPT code -> case count
    complication_rate: float                        # 0.0 - 1.0
    avg_rating: float = 5.0                        # 1-5 peer rating
    available_slots: list[str] = field(default_factory=list)  # ISO datetime strings
    hourly_rate_usd: float = 0.0

    def case_count_for(self, cpt_code: str) -> int:
        return self.procedure_log.get(cpt_code, 0)


@dataclass
class SurgicalCase:
    id: str
    patient_age: int
    procedure: Procedure
    additional_procedures: list[Procedure] = field(default_factory=list)  # combined cases
    hospital_state: str = ""
    hospital_device: DeviceType = DeviceType.DAVINCI_XI
    requested_datetime: str = ""          # ISO datetime
    network_latency_ms: float = 0.0       # measured latency to hospital
    budget_usd: float = float("inf")


@dataclass
class MatchResult:
    surgeon: SurgeonProfile
    score: float
    breakdown: dict[str, float]
    disqualified: bool = False
    disqualify_reason: str = ""


# =============================================================================
# 4. MATCHING ENGINE
# =============================================================================

class MatchingEngine:

    # --- Scoring Weights (tunable) ---
    WEIGHT_CASE_VOLUME = 0.40
    WEIGHT_COMPLICATION = 0.25
    WEIGHT_SUBSPECIALTY = 0.20
    WEIGHT_RATING = 0.15

    def match(self, case: SurgicalCase, surgeons: list[SurgeonProfile]) -> list[MatchResult]:
        """
        Run all three layers and return a ranked list of MatchResults.
        """
        results = []

        for surgeon in surgeons:
            # --- LAYER 1: Hard Filter ---
            passed, reason = self._hard_filter(case, surgeon)
            if not passed:
                results.append(MatchResult(
                    surgeon=surgeon,
                    score=0.0,
                    breakdown={},
                    disqualified=True,
                    disqualify_reason=reason,
                ))
                continue

            # --- LAYER 2: Quality Score ---
            score, breakdown = self._quality_score(case, surgeon)

            # --- LAYER 3: Feasibility Adjustment ---
            score, breakdown = self._feasibility_adjust(case, surgeon, score, breakdown)

            results.append(MatchResult(
                surgeon=surgeon,
                score=round(score, 2),
                breakdown={k: round(v, 2) for k, v in breakdown.items()},
            ))

        # Sort: qualified first (by score desc), then disqualified
        results.sort(key=lambda r: (not r.disqualified, r.score), reverse=True)
        return results

    # -----------------------------------------------------------------
    # LAYER 1: Hard Filter
    # -----------------------------------------------------------------
    def _hard_filter(self, case: SurgicalCase, surgeon: SurgeonProfile) -> tuple[bool, str]:

        # 1a. State license
        if case.hospital_state and case.hospital_state not in surgeon.licensed_states:
            return False, f"No license in {case.hospital_state}"

        # 1b. Device certification
        if case.hospital_device not in surgeon.certified_devices:
            return False, f"Not certified on {case.hospital_device.value}"

        # 1c. Availability
        if case.requested_datetime and surgeon.available_slots:
            if case.requested_datetime not in surgeon.available_slots:
                return False, f"Not available at {case.requested_datetime}"

        # 1d. Must have done the primary procedure at least once
        if surgeon.case_count_for(case.procedure.code) == 0:
            return False, f"No experience with {case.procedure.code} ({case.procedure.name})"

        return True, ""

    # -----------------------------------------------------------------
    # LAYER 2: Quality Score (0 - 100)
    # -----------------------------------------------------------------
    def _quality_score(self, case: SurgicalCase, surgeon: SurgeonProfile) -> tuple[float, dict]:

        breakdown = {}

        # 2a. Case volume score (log scale, cap at 500)
        all_codes = [case.procedure.code] + [p.code for p in case.additional_procedures]
        total_cases = sum(surgeon.case_count_for(c) for c in all_codes)
        volume_score = min(total_cases / 500, 1.0) * 100
        breakdown["case_volume"] = volume_score

        # 2b. Complication rate score (lower is better)
        complication_score = (1.0 - surgeon.complication_rate) * 100
        breakdown["complication_rate"] = complication_score

        # 2c. Subspecialty match depth
        required_subs = {case.procedure.subspecialty}
        for p in case.additional_procedures:
            required_subs.add(p.subspecialty)
        matched = required_subs.intersection(set(surgeon.subspecialties))
        sub_score = (len(matched) / len(required_subs)) * 100 if required_subs else 100
        breakdown["subspecialty_match"] = sub_score

        # 2d. Peer rating
        rating_score = (surgeon.avg_rating / 5.0) * 100
        breakdown["peer_rating"] = rating_score

        # Weighted total
        total = (
            self.WEIGHT_CASE_VOLUME * volume_score
            + self.WEIGHT_COMPLICATION * complication_score
            + self.WEIGHT_RATING * rating_score
            + self.WEIGHT_SUBSPECIALTY * sub_score
        )

        return total, breakdown

    # -----------------------------------------------------------------
    # LAYER 3: Feasibility Adjustment
    # -----------------------------------------------------------------
    def _feasibility_adjust(
        self,
        case: SurgicalCase,
        surgeon: SurgeonProfile,
        score: float,
        breakdown: dict,
    ) -> tuple[float, dict]:

        # 3a. Network latency penalty
        if case.network_latency_ms > MAX_SAFE_LATENCY_MS:
            penalty = min((case.network_latency_ms - MAX_SAFE_LATENCY_MS) / 100, 1.0) * 30
            score -= penalty
            breakdown["latency_penalty"] = -penalty

        # 3b. Budget filter
        if surgeon.hourly_rate_usd > case.budget_usd:
            over = (surgeon.hourly_rate_usd - case.budget_usd) / case.budget_usd
            penalty = min(over, 1.0) * 20
            score -= penalty
            breakdown["budget_penalty"] = -penalty

        # 3c. Complexity bonus — reward surgeons with HIGH volume for HIGH complexity cases
        if case.procedure.complexity.value >= 3:
            primary_count = surgeon.case_count_for(case.procedure.code)
            if primary_count >= 200:
                bonus = 10
                score += bonus
                breakdown["complexity_bonus"] = bonus

        return score, breakdown


# =============================================================================
# 5. DEMO
# =============================================================================

def run_demo():
    print("=" * 70)
    print("  TELESURGERY MATCHING FRAMEWORK — DEMO")
    print("=" * 70)

    # --- Define surgeons ---
    surgeons = [
        SurgeonProfile(
            id="S001",
            name="Dr. Alice Chen",
            specialty=Specialty.OPHTHALMOLOGY,
            subspecialties=["Glaucoma", "Cataract"],
            licensed_states=["CA", "NY", "TX", "FL"],
            certified_devices=[DeviceType.DAVINCI_XI, DeviceType.DAVINCI_SP],
            procedure_log={"66711": 320, "66984": 800, "66987": 150, "66170": 90},
            complication_rate=0.02,
            avg_rating=4.9,
            available_slots=["2026-05-10T08:00"],
            hourly_rate_usd=600,
        ),
        SurgeonProfile(
            id="S002",
            name="Dr. Bob Martinez",
            specialty=Specialty.OPHTHALMOLOGY,
            subspecialties=["Retina"],
            licensed_states=["CA", "TX"],
            certified_devices=[DeviceType.DAVINCI_XI],
            procedure_log={"67036": 500, "67108": 200, "66984": 50},
            complication_rate=0.03,
            avg_rating=4.7,
            available_slots=["2026-05-10T08:00"],
            hourly_rate_usd=700,
        ),
        SurgeonProfile(
            id="S003",
            name="Dr. Carol Wu",
            specialty=Specialty.OPHTHALMOLOGY,
            subspecialties=["Glaucoma", "Cataract"],
            licensed_states=["NY", "NJ", "CT"],
            certified_devices=[DeviceType.HUGO_RAS],
            procedure_log={"66711": 50, "66984": 200, "66987": 30},
            complication_rate=0.05,
            avg_rating=4.3,
            available_slots=["2026-05-10T08:00"],
            hourly_rate_usd=450,
        ),
        SurgeonProfile(
            id="S004",
            name="Dr. David Park",
            specialty=Specialty.OPHTHALMOLOGY,
            subspecialties=["Glaucoma"],
            licensed_states=["CA", "TX", "FL", "NY"],
            certified_devices=[DeviceType.DAVINCI_XI, DeviceType.HUGO_RAS],
            procedure_log={"66711": 180, "66170": 300, "66183": 120, "66984": 10},
            complication_rate=0.04,
            avg_rating=4.6,
            available_slots=["2026-05-10T08:00"],
            hourly_rate_usd=550,
        ),
    ]

    # --- Define case ---
    # 72-year-old, open-angle glaucoma + early cataract → MIGS + cataract combined
    case = SurgicalCase(
        id="CASE-2026-0042",
        patient_age=72,
        procedure=PROCEDURES["66711"],            # MIGS
        additional_procedures=[PROCEDURES["66984"]],  # + standard cataract
        hospital_state="CA",
        hospital_device=DeviceType.DAVINCI_XI,
        requested_datetime="2026-05-10T08:00",
        network_latency_ms=85,
        budget_usd=700,
    )

    print(f"\n  CASE: {case.id}")
    print(f"  Patient: {case.patient_age}yo")
    print(f"  Primary: {case.procedure.name} ({case.procedure.code})")
    for p in case.additional_procedures:
        print(f"  + Combined: {p.name} ({p.code})")
    print(f"  Location: {case.hospital_state}  |  Device: {case.hospital_device.value}")
    print(f"  Latency: {case.network_latency_ms}ms  |  Budget: ${case.budget_usd}/hr")
    print("-" * 70)

    # --- Run matching ---
    engine = MatchingEngine()
    results = engine.match(case, surgeons)

    print(f"\n  RESULTS (ranked)\n")
    for i, r in enumerate(results, 1):
        status = "DISQUALIFIED" if r.disqualified else f"Score: {r.score}"
        print(f"  #{i}  {r.surgeon.name:<22}  {status}")
        if r.disqualified:
            print(f"      Reason: {r.disqualify_reason}")
        else:
            for k, v in r.breakdown.items():
                label = k.replace("_", " ").title()
                print(f"      {label:<22} {v:>+7.1f}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    run_demo()
