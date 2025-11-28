"""
Functional Taxonomy for Cryptocurrency Projects

Defines the categories of functional claims that crypto projects make
in their whitepapers and communications. These categories form the
feature dimension of the claims tensor.

Based on literature review and crypto market structure analysis.
Designed for zero-shot classification with BART MNLI.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class FunctionalCategory(Enum):
    """Primary functional categories for crypto projects."""
    
    STORE_OF_VALUE = "store_of_value"
    MEDIUM_OF_EXCHANGE = "medium_of_exchange"
    SMART_CONTRACTS = "smart_contracts"
    INFRASTRUCTURE = "infrastructure"
    PRIVACY = "privacy"
    GOVERNANCE = "governance"
    DATA_ORACLE = "data_oracle"
    IDENTITY = "identity"
    GAMING_METAVERSE = "gaming_metaverse"
    STABLECOIN = "stablecoin"


@dataclass
class TaxonomyEntry:
    """Definition of a functional category with classification hints."""
    
    category: FunctionalCategory
    description: str
    keywords: List[str]
    zero_shot_hypothesis: str  # For BART MNLI zero-shot classification
    subcategories: List[str] = field(default_factory=list)
    
    def get_candidate_labels(self) -> List[str]:
        """Get labels for zero-shot classification."""
        return [self.description] + self.subcategories


# Core functional taxonomy
# Each entry designed for zero-shot classification compatibility
FUNCTIONAL_TAXONOMY: Dict[FunctionalCategory, TaxonomyEntry] = {
    
    FunctionalCategory.STORE_OF_VALUE: TaxonomyEntry(
        category=FunctionalCategory.STORE_OF_VALUE,
        description="digital store of value and savings",
        keywords=[
            "store of value", "digital gold", "savings", "wealth preservation",
            "inflation hedge", "hard money", "scarcity", "fixed supply",
            "deflationary", "sound money", "monetary policy", "reserve asset"
        ],
        zero_shot_hypothesis="This text is about storing value or preserving wealth.",
        subcategories=[
            "digital gold",
            "inflation hedge", 
            "long-term savings",
            "reserve asset"
        ]
    ),
    
    FunctionalCategory.MEDIUM_OF_EXCHANGE: TaxonomyEntry(
        category=FunctionalCategory.MEDIUM_OF_EXCHANGE,
        description="payment system and medium of exchange",
        keywords=[
            "payments", "transactions", "remittances", "micropayments",
            "peer-to-peer", "transfer", "currency", "cash", "spending",
            "merchant", "point of sale", "payment network", "fast transactions"
        ],
        zero_shot_hypothesis="This text is about making payments or transferring money.",
        subcategories=[
            "peer-to-peer payments",
            "cross-border remittances",
            "micropayments",
            "merchant payments"
        ]
    ),
    
    FunctionalCategory.SMART_CONTRACTS: TaxonomyEntry(
        category=FunctionalCategory.SMART_CONTRACTS,
        description="programmable smart contracts and decentralized applications",
        keywords=[
            "smart contracts", "dapps", "programmable", "turing complete",
            "defi", "lending", "borrowing", "yield", "liquidity",
            "automated market maker", "amm", "dex", "derivatives",
            "nft", "tokenization", "composability"
        ],
        zero_shot_hypothesis="This text is about smart contracts or decentralized applications.",
        subcategories=[
            "decentralized finance (DeFi)",
            "lending and borrowing",
            "decentralized exchanges",
            "NFTs and tokenization"
        ]
    ),
    
    FunctionalCategory.INFRASTRUCTURE: TaxonomyEntry(
        category=FunctionalCategory.INFRASTRUCTURE,
        description="blockchain infrastructure and scaling solutions",
        keywords=[
            "scaling", "layer 2", "layer 1", "throughput", "tps",
            "interoperability", "bridge", "cross-chain", "sidechain",
            "rollup", "sharding", "consensus", "proof of stake",
            "proof of work", "validator", "node", "network"
        ],
        zero_shot_hypothesis="This text is about blockchain infrastructure or scaling.",
        subcategories=[
            "layer 2 scaling",
            "cross-chain interoperability",
            "consensus mechanisms",
            "network infrastructure"
        ]
    ),
    
    FunctionalCategory.PRIVACY: TaxonomyEntry(
        category=FunctionalCategory.PRIVACY,
        description="privacy-preserving transactions and confidential computing",
        keywords=[
            "privacy", "anonymous", "confidential", "zero knowledge",
            "zk-snark", "zk-stark", "ring signature", "mixer",
            "stealth address", "encrypted", "untraceable", "fungible"
        ],
        zero_shot_hypothesis="This text is about privacy or anonymous transactions.",
        subcategories=[
            "anonymous transactions",
            "zero-knowledge proofs",
            "confidential computing",
            "privacy-preserving protocols"
        ]
    ),
    
    FunctionalCategory.GOVERNANCE: TaxonomyEntry(
        category=FunctionalCategory.GOVERNANCE,
        description="decentralized governance and DAOs",
        keywords=[
            "governance", "dao", "voting", "proposal", "treasury",
            "delegation", "quadratic voting", "token voting",
            "community", "decentralized autonomous", "council"
        ],
        zero_shot_hypothesis="This text is about governance or decentralized decision-making.",
        subcategories=[
            "DAOs",
            "on-chain voting",
            "treasury management",
            "governance tokens"
        ]
    ),
    
    FunctionalCategory.DATA_ORACLE: TaxonomyEntry(
        category=FunctionalCategory.DATA_ORACLE,
        description="data oracles and external data feeds",
        keywords=[
            "oracle", "data feed", "price feed", "external data",
            "off-chain", "api", "real world data", "chainlink",
            "decentralized oracle", "data provider"
        ],
        zero_shot_hypothesis="This text is about data oracles or external data feeds.",
        subcategories=[
            "price oracles",
            "decentralized data feeds",
            "real-world data integration"
        ]
    ),
    
    FunctionalCategory.IDENTITY: TaxonomyEntry(
        category=FunctionalCategory.IDENTITY,
        description="digital identity and authentication",
        keywords=[
            "identity", "did", "decentralized identity", "ssi",
            "self-sovereign", "credential", "verification", "kyc",
            "authentication", "soulbound", "reputation"
        ],
        zero_shot_hypothesis="This text is about digital identity or authentication.",
        subcategories=[
            "decentralized identity (DID)",
            "self-sovereign identity",
            "credential verification",
            "reputation systems"
        ]
    ),
    
    FunctionalCategory.GAMING_METAVERSE: TaxonomyEntry(
        category=FunctionalCategory.GAMING_METAVERSE,
        description="gaming, metaverse, and virtual worlds",
        keywords=[
            "gaming", "game", "metaverse", "virtual world", "play to earn",
            "gamefi", "virtual reality", "avatar", "land", "in-game",
            "esports", "entertainment"
        ],
        zero_shot_hypothesis="This text is about gaming or virtual worlds.",
        subcategories=[
            "play-to-earn gaming",
            "metaverse platforms",
            "virtual real estate",
            "in-game assets"
        ]
    ),
    
    FunctionalCategory.STABLECOIN: TaxonomyEntry(
        category=FunctionalCategory.STABLECOIN,
        description="stablecoins and price-stable assets",
        keywords=[
            "stablecoin", "stable", "pegged", "dollar", "fiat-backed",
            "algorithmic", "collateralized", "reserve", "peg",
            "price stability", "usd", "usdt", "usdc"
        ],
        zero_shot_hypothesis="This text is about stablecoins or price-stable assets.",
        subcategories=[
            "fiat-backed stablecoins",
            "algorithmic stablecoins",
            "collateralized stablecoins"
        ]
    ),
}


def get_all_keywords() -> List[str]:
    """Get all keywords across all categories."""
    keywords = []
    for entry in FUNCTIONAL_TAXONOMY.values():
        keywords.extend(entry.keywords)
    return list(set(keywords))


def get_zero_shot_labels() -> Dict[str, FunctionalCategory]:
    """Get mapping of zero-shot labels to categories."""
    labels = {}
    for category, entry in FUNCTIONAL_TAXONOMY.items():
        labels[entry.description] = category
        for subcat in entry.subcategories:
            labels[subcat] = category
    return labels


def get_candidate_labels_flat() -> List[str]:
    """Get flat list of all candidate labels for zero-shot classification."""
    labels = []
    for entry in FUNCTIONAL_TAXONOMY.values():
        labels.append(entry.description)
        labels.extend(entry.subcategories)
    return labels


def category_to_vector(categories: List[FunctionalCategory]) -> List[float]:
    """Convert list of categories to one-hot vector."""
    all_categories = list(FunctionalCategory)
    vector = [0.0] * len(all_categories)
    for cat in categories:
        idx = all_categories.index(cat)
        vector[idx] = 1.0
    return vector


def vector_to_categories(vector: List[float], threshold: float = 0.5) -> List[FunctionalCategory]:
    """Convert probability vector back to category list."""
    all_categories = list(FunctionalCategory)
    return [
        all_categories[i] 
        for i, prob in enumerate(vector) 
        if prob >= threshold
    ]


# Top 20 crypto projects for PoC (by market cap, Nov 2025)
# Selected for: whitepaper availability, Binance listing, 1+ year history
TOP_20_PROJECTS = [
    {"symbol": "BTC", "name": "Bitcoin", "whitepaper": "bitcoin.pdf"},
    {"symbol": "ETH", "name": "Ethereum", "whitepaper": "ethereum.pdf"},
    {"symbol": "BNB", "name": "BNB Chain", "whitepaper": "bnb.pdf"},
    {"symbol": "SOL", "name": "Solana", "whitepaper": "solana.pdf"},
    {"symbol": "XRP", "name": "Ripple", "whitepaper": "ripple.pdf"},
    {"symbol": "DOGE", "name": "Dogecoin", "whitepaper": None},  # No formal whitepaper
    {"symbol": "ADA", "name": "Cardano", "whitepaper": "cardano.pdf"},
    {"symbol": "AVAX", "name": "Avalanche", "whitepaper": "avalanche.pdf"},
    {"symbol": "TRX", "name": "TRON", "whitepaper": "tron.pdf"},
    {"symbol": "LINK", "name": "Chainlink", "whitepaper": "chainlink.pdf"},
    {"symbol": "DOT", "name": "Polkadot", "whitepaper": "polkadot.pdf"},
    {"symbol": "MATIC", "name": "Polygon", "whitepaper": "polygon.pdf"},
    {"symbol": "LTC", "name": "Litecoin", "whitepaper": "litecoin.pdf"},
    {"symbol": "UNI", "name": "Uniswap", "whitepaper": "uniswap.pdf"},
    {"symbol": "ATOM", "name": "Cosmos", "whitepaper": "cosmos.pdf"},
    {"symbol": "XLM", "name": "Stellar", "whitepaper": "stellar.pdf"},
    {"symbol": "XMR", "name": "Monero", "whitepaper": "monero.pdf"},
    {"symbol": "NEAR", "name": "NEAR Protocol", "whitepaper": "near.pdf"},
    {"symbol": "ALGO", "name": "Algorand", "whitepaper": "algorand.pdf"},
    {"symbol": "FIL", "name": "Filecoin", "whitepaper": "filecoin.pdf"},
]


if __name__ == "__main__":
    print("=== Functional Taxonomy for Crypto Projects ===\n")
    
    for category, entry in FUNCTIONAL_TAXONOMY.items():
        print(f"\n{category.value.upper()}")
        print(f"  Description: {entry.description}")
        print(f"  Zero-shot: \"{entry.zero_shot_hypothesis}\"")
        print(f"  Keywords: {', '.join(entry.keywords[:5])}...")
        print(f"  Subcategories: {entry.subcategories}")
    
    print(f"\n\nTotal categories: {len(FUNCTIONAL_TAXONOMY)}")
    print(f"Total keywords: {len(get_all_keywords())}")
    print(f"Total candidate labels: {len(get_candidate_labels_flat())}")
    
    print("\n\n=== Top 20 Projects for PoC ===")
    for proj in TOP_20_PROJECTS:
        wp_status = "✓" if proj["whitepaper"] else "✗"
        print(f"  {wp_status} {proj['symbol']}: {proj['name']}")
