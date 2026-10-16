"""Conservative, auditable V4 gate, not a general natural-language entailment model.

Only a complete standalone line in a cited chunk can qualify. Its entire text
must use the grammar documented in POLICY. Unrecognized prose fails closed;
we neither annotate evidence nor infer relationships from generic vocabulary.
The canonical profile exposes training-relative bins, not raw measurements.
"""
import re

from src.rag import phase2_common as common

BINS = ("extremely high", "very high", "high", "moderately high", "typical",
        "moderately low", "low", "very low", "extremely low")
CLASS_TEXT = {"benign": "benign", "malware": "malware", "Tor": "Tor", "VPN": "VPN",
              "Non-Tor": "non-Tor", "NonVPN": "non-VPN"}
POLICY = (
    "V4 positive gate is fail-closed and recognizes only this complete standalone evidence line: "
    "'<Bin> <feature> relative to the training distribution indicates <class> traffic.' "
    "Bin must be one of extremely high, very high, high, moderately high, typical, moderately low, low, very low, extremely low "
    "(first letter capitalized). Feature and bin must exactly match an observed '<feature> is <bin>' entry "
    "in the supplied training-relative traffic profile; no aliases, numeric conversions or inferred directions. "
    "Class must exactly name the supported allowed class (benign, malware, Tor, VPN, non-Tor or non-VPN). "
    "The quoted line must occur verbatim as a complete line in a cited supplied chunk. "
    "No rewriting evidence into this grammar is permitted. Unrecognized or ambiguous prose fails closed. "
    "EVIDENCE_LINK must explicitly include the matched '<feature> is <bin>' observation. "
    "This intentionally incomplete grammar cannot establish general semantic entailment; human grounding review remains required."
)
PROFILE_START = ". Observed traffic profile relative to the training distribution: "
PROFILE_END = ". Retrieve trusted cybersecurity evidence relevant to the predicted class and these observed traffic characteristics."


def validate(value, prompt):
    if value["evidence_status"] not in {"SUPPORTED", "CONTRADICTED"}:
        return
    match = re.fullmatch(r'EVIDENCE_QUOTE: "([^"\r\n]+)"\nEVIDENCE_LINK: ([^\r\n]+)', value["explanation"])
    common.require(match is not None, "V4 positive response requires nonempty EVIDENCE_QUOTE and EVIDENCE_LINK")
    quote, link = match.groups()
    common.require(quote.strip() and link.strip(), "Empty evidence quote/link")
    payload = common.parse_json(prompt["prompt_text"].partition("INPUT_JSON:\n")[2])
    cited = [item["chunk_text"] for item in payload["trusted_evidence"] if item["chunk_id"] in value["cited_chunk_ids"]]
    common.require(any(quote in text for text in cited), "Evidence quote must occur verbatim in a cited supplied chunk")
    common.require(any(quote in text.splitlines() for text in cited), "Qualifying quote must be a complete standalone evidence line")
    profile = payload["traffic_profile"]
    common.require(isinstance(profile, str) and profile.count(PROFILE_START) == 1 and profile.endswith(PROFILE_END), "Unrecognized observed profile; positive evidence cannot qualify")
    observations = profile.split(PROFILE_START, 1)[1][:-len(PROFILE_END)].split("; ")
    qualified = []
    for observation in observations:
        feature, separator, direction = observation.rpartition(" is ")
        if not separator or direction not in BINS or re.fullmatch(r"[A-Za-z0-9 /-]+", feature) is None:
            continue
        # Only the observed part of the profile participates, never prediction/confidence.
        expected = f"{direction.capitalize()} {feature} relative to the training distribution indicates {CLASS_TEXT[value['supported_class']]} traffic."
        if quote == expected and observation in link:
            qualified.append(observation)
    common.require(len(qualified) == 1, "Quote does not establish an exact observed feature/direction to supported-class relationship")
