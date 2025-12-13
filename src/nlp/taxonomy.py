"""
Functional claim taxonomy for cryptocurrency whitepaper classification.

10 categories derived from cryptocurrency whitepaper analysis covering
the primary functional claims made by blockchain projects.
"""

FUNCTIONAL_CATEGORIES = {
    "store_of_value": {
        "label": "Store of Value",
        "description": "Digital gold, inflation hedge, wealth preservation",
        "keywords": ["store of value", "digital gold", "inflation hedge", "scarcity",
                    "wealth preservation", "hard money", "fixed supply", "deflation"]
    },
    "medium_of_exchange": {
        "label": "Medium of Exchange",
        "description": "Payment system, transactions, currency",
        "keywords": ["payment", "transaction", "currency", "transfer", "remittance",
                    "peer-to-peer", "micropayment", "cash", "money"]
    },
    "smart_contracts": {
        "label": "Smart Contracts",
        "description": "Programmable contracts, automation, trustless execution",
        "keywords": ["smart contract", "programmable", "automation", "trustless",
                    "self-executing", "conditional logic", "state machine"]
    },
    "defi": {
        "label": "Decentralized Finance",
        "description": "Lending, borrowing, yield, liquidity provision",
        "keywords": ["defi", "lending", "borrowing", "yield", "liquidity", "amm",
                    "automated market maker", "swap", "collateral", "staking"]
    },
    "governance": {
        "label": "Governance",
        "description": "Voting, DAOs, community decision-making",
        "keywords": ["governance", "voting", "dao", "proposal", "delegate",
                    "community", "decision", "protocol upgrade", "treasury"]
    },
    "scalability": {
        "label": "Scalability",
        "description": "High throughput, low latency, Layer 2 solutions",
        "keywords": ["scalability", "throughput", "tps", "latency", "layer 2",
                    "rollup", "sharding", "parallel", "performance"]
    },
    "privacy": {
        "label": "Privacy",
        "description": "Anonymous transactions, zero-knowledge proofs",
        "keywords": ["privacy", "anonymous", "zero-knowledge", "zkp", "confidential",
                    "ring signature", "stealth", "mixer", "obfuscation"]
    },
    "interoperability": {
        "label": "Interoperability",
        "description": "Cross-chain communication, bridges, multi-chain",
        "keywords": ["interoperability", "cross-chain", "bridge", "multi-chain",
                    "relay", "atomic swap", "ibc", "cosmos sdk"]
    },
    "data_storage": {
        "label": "Data Storage",
        "description": "Decentralized storage, file systems, permanence",
        "keywords": ["storage", "file", "data", "ipfs", "permanent", "archive",
                    "decentralized storage", "distributed", "content addressing"]
    },
    "oracle": {
        "label": "Oracle Services",
        "description": "External data feeds, real-world information",
        "keywords": ["oracle", "data feed", "external data", "api", "off-chain",
                    "real-world", "price feed", "verifiable randomness"]
    }
}

# Zero-shot classification labels (for BART-MNLI)
ZERO_SHOT_LABELS = [
    "This text describes a store of value or digital gold.",
    "This text describes a payment system or medium of exchange.",
    "This text describes smart contracts or programmable execution.",
    "This text describes decentralized finance, lending, or yield.",
    "This text describes governance, voting, or DAOs.",
    "This text describes scalability, throughput, or performance.",
    "This text describes privacy or anonymous transactions.",
    "This text describes cross-chain interoperability or bridges.",
    "This text describes decentralized data storage.",
    "This text describes oracle services or external data feeds."
]

# Mapping from zero-shot index to category key
LABEL_TO_CATEGORY = [
    "store_of_value",
    "medium_of_exchange",
    "smart_contracts",
    "defi",
    "governance",
    "scalability",
    "privacy",
    "interoperability",
    "data_storage",
    "oracle"
]


def get_category_names() -> list[str]:
    """Return ordered list of category keys."""
    return LABEL_TO_CATEGORY.copy()


def get_category_labels() -> list[str]:
    """Return human-readable category labels."""
    return [FUNCTIONAL_CATEGORIES[cat]["label"] for cat in LABEL_TO_CATEGORY]
