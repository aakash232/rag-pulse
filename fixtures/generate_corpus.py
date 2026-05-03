"""
Generate the synthetic 50-chunk benchmark corpus.

Run with:  uv run python benchmark/generate_corpus.py

Corpus structure (50 chunks across 3 collections):
  docs/    20 chunks — refund policy, support hours, privacy, time-bounded promos, orphans
  pricing/ 15 chunks — evolving starter/pro/enterprise pricing (numeric contradictions)
  api/     15 chunks — SDK version supersession, rate-limit contradiction, stable methods
"""

import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(42)
DIM = 384  # sentence-transformers/all-MiniLM-L6-v2 dimension


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _base() -> np.ndarray:
    return _unit(RNG.standard_normal(DIM))


def _near_dup(base: np.ndarray, noise: float = 0.02) -> np.ndarray:
    """Cosine similarity to base ≈ 0.98–0.999 (lexical/semantic near-duplicate)."""
    return _unit(base + RNG.standard_normal(DIM) * noise)


def _cluster_member(base: np.ndarray, noise: float = 0.18) -> np.ndarray:
    """Cosine similarity to base ≈ 0.82–0.94 (same topic cluster)."""
    return _unit(base + RNG.standard_normal(DIM) * noise)


def _orphan() -> np.ndarray:
    """Completely random unit vector — no cluster affiliation."""
    return _unit(RNG.standard_normal(DIM))


def emb(v: np.ndarray) -> list[float]:
    return v.astype(float).tolist()


# ---------------------------------------------------------------------------
# Topic cluster base vectors (generated once, deterministic via RNG seed)
# ---------------------------------------------------------------------------

B = {
    k: _base()
    for k in [
        "refund",
        "support",
        "privacy",
        "promo",
        "p_starter",
        "p_pro",
        "p_enterprise",
        "p_misc",
        "api_install",
        "api_auth",
        "api_methods",
    ]
}


# ---------------------------------------------------------------------------
# Corpus definitions
# ---------------------------------------------------------------------------


def docs_collection() -> list[dict]:
    r = B["refund"]
    s = B["support"]
    pr = B["promo"]

    return [
        # --- Refund policy cluster (A) ---
        # A1: canonical
        {
            "id": "docs-refund-001",
            "text": (
                "Returns and refunds are accepted within 30 days of purchase. "
                "Items must be in original condition. Refunds are processed to the "
                "original payment method within 5-7 business days."
            ),
            "embedding": emb(_near_dup(r, noise=0.001)),  # canonical — very clean
            "metadata": {"created_at": "2023-01-10T09:00:00Z", "source": "help-center/refunds.md"},
        },
        # A2: lexical near-duplicate of A1 (formatting differences)
        {
            "id": "docs-refund-002",
            "text": (
                "Refunds accepted within 30 days from date of purchase. "
                "Items must be returned in their original condition. "
                "Refunds are typically processed to the original payment method within 5 to 7 business days."
            ),
            "embedding": emb(_near_dup(r, noise=0.018)),  # very close to A1
            "metadata": {"created_at": "2023-01-12T14:00:00Z", "source": "faq/billing.md"},
        },
        # A3: semantic near-duplicate of A1 (paraphrase)
        {
            "id": "docs-refund-003",
            "text": (
                "Our money-back guarantee lets customers return purchases within one month. "
                "Returned items should be unused and in their original packaging. "
                "Credit appears on your statement within a week."
            ),
            "embedding": emb(_near_dup(r, noise=0.025)),  # semantic dup
            "metadata": {"created_at": "2023-02-01T10:00:00Z", "source": "policies/returns.md"},
        },
        # A4: same cluster, NOT a dup (specific sub-policy)
        {
            "id": "docs-refund-004",
            "text": (
                "To initiate a refund, submit a request through the support portal at "
                "support.example.com/refunds. Include your order number and reason for return."
            ),
            "embedding": emb(_cluster_member(r, noise=0.20)),
            "metadata": {"created_at": "2023-01-15T11:00:00Z", "source": "help-center/refunds.md"},
        },
        # --- Support hours cluster (B) ---
        # B1: canonical
        {
            "id": "docs-support-001",
            "text": (
                "Customer support is available Monday through Friday, "
                "9am to 6pm Eastern Time. Response time for standard tickets is 24 hours."
            ),
            "embedding": emb(_near_dup(s, noise=0.001)),
            "metadata": {"created_at": "2023-01-10T09:00:00Z", "source": "help-center/contact.md"},
        },
        # B2: lexical near-duplicate of B1
        {
            "id": "docs-support-002",
            "text": (
                "Our support team operates Monday–Friday from 9:00 AM to 6:00 PM Eastern Time. "
                "Standard ticket response time: 24 hours."
            ),
            "embedding": emb(_near_dup(s, noise=0.019)),
            "metadata": {"created_at": "2023-03-05T16:00:00Z", "source": "landing-page/support.md"},
        },
        # B3: same cluster, different content
        {
            "id": "docs-support-003",
            "text": (
                "For critical production outages outside business hours, "
                "use the emergency hotline: +1-800-555-0199. Available 24/7 for Pro and Enterprise customers."
            ),
            "embedding": emb(_cluster_member(s, noise=0.22)),
            "metadata": {"created_at": "2023-01-10T09:00:00Z", "source": "help-center/contact.md"},
        },
        # --- Privacy cluster (C) ---
        {
            "id": "docs-privacy-001",
            "text": (
                "We collect only the data necessary to provide our services, "
                "as described in our Privacy Policy. Data is retained for no longer than 3 years."
            ),
            "embedding": emb(_near_dup(B["privacy"], noise=0.001)),
            "metadata": {"created_at": "2022-11-01T00:00:00Z", "source": "legal/privacy.md"},
        },
        {
            "id": "docs-privacy-002",
            "text": "Your data is encrypted in transit using TLS 1.3 and at rest using AES-256.",
            "embedding": emb(_cluster_member(B["privacy"], noise=0.16)),
            "metadata": {"created_at": "2022-11-01T00:00:00Z", "source": "legal/privacy.md"},
        },
        {
            "id": "docs-privacy-003",
            "text": "We do not sell, rent, or share user data with third parties for marketing purposes.",
            "embedding": emb(_cluster_member(B["privacy"], noise=0.17)),
            "metadata": {"created_at": "2022-11-01T00:00:00Z", "source": "legal/privacy.md"},
        },
        {
            "id": "docs-privacy-004",
            "text": (
                "Users may request deletion of their account and all associated data "
                "by contacting privacy@example.com. Requests are fulfilled within 30 days."
            ),
            "embedding": emb(_cluster_member(B["privacy"], noise=0.19)),
            "metadata": {"created_at": "2022-11-01T00:00:00Z", "source": "legal/privacy.md"},
        },
        # --- Time-bounded content (D) — stale by content, not corpus dynamics ---
        {
            "id": "docs-promo-001",
            "text": (
                "Spring promotion: Get 20% off all annual plans through March 31, 2023. Use code SPRING23 at checkout."
            ),
            "embedding": emb(_cluster_member(pr, noise=0.12)),
            "metadata": {"created_at": "2023-03-01T00:00:00Z", "source": "marketing/promos.md"},
        },
        {
            "id": "docs-promo-002",
            "text": (
                "Limited-time offer: Free onboarding session (value $500) for new annual subscribers "
                "through December 31, 2022."
            ),
            "embedding": emb(_cluster_member(pr, noise=0.14)),
            "metadata": {"created_at": "2022-11-15T00:00:00Z", "source": "marketing/promos.md"},
        },
        {
            "id": "docs-promo-003",
            "text": ("Holiday special: Sign up before January 1, 2023 and receive 3 months free on any paid plan."),
            "embedding": emb(_cluster_member(pr, noise=0.13)),
            "metadata": {"created_at": "2022-12-01T00:00:00Z", "source": "marketing/promos.md"},
        },
        # --- Isolated orphan chunks (E) — diverse topics, no cluster ---
        {
            "id": "docs-orphan-001",
            "text": "The company was founded in 2018 and is headquartered in San Francisco, CA.",
            "embedding": emb(_orphan()),
            "metadata": {"created_at": "2022-06-01T00:00:00Z", "source": "about/company.md"},
        },
        {
            "id": "docs-orphan-002",
            "text": "Contact our sales team at sales@example.com for volume licensing and enterprise inquiries.",
            "embedding": emb(_orphan()),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "contact/sales.md"},
        },
        {
            "id": "docs-orphan-003",
            "text": "Compatible with macOS 12 Monterey and later, Windows 10+, and Ubuntu 20.04 LTS.",
            "embedding": emb(_orphan()),
            "metadata": {
                "created_at": "2022-09-01T00:00:00Z",
                "source": "docs/system-requirements.md",
            },
        },
        {
            "id": "docs-orphan-004",
            "text": (
                "Awards: G2 Leader in Project Management (Summer 2024), "
                "Capterra Best Value 2023, TrustRadius Top Rated 2024."
            ),
            "embedding": emb(_orphan()),
            "metadata": {"created_at": "2024-07-01T00:00:00Z", "source": "about/awards.md"},
        },
        {
            "id": "docs-orphan-005",
            "text": ("The platform processes approximately 10 million events per day across all customer tenants."),
            "embedding": emb(_orphan()),
            "metadata": {"created_at": "2023-06-01T00:00:00Z", "source": "docs/scale.md"},
        },
        {
            "id": "docs-orphan-006",
            "text": "Keyboard shortcut reference: Cmd+K opens the command palette. Cmd+/ toggles comments.",
            "embedding": emb(_orphan()),
            "metadata": {
                "created_at": "2023-04-01T00:00:00Z",
                "source": "docs/keyboard-shortcuts.md",
            },
        },
    ]


def pricing_collection() -> list[dict]:
    ps = B["p_starter"]
    pp = B["p_pro"]
    pe = B["p_enterprise"]
    pm = B["p_misc"]

    return [
        # --- Starter plan: numeric price contradiction ---
        # PS1: OLD price (Jan 2023)
        {
            "id": "pricing-starter-001",
            "text": (
                "Starter plan: $29/month (billed monthly) or $23/month (billed annually). "
                "Includes up to 5 users and 10GB storage."
            ),
            "embedding": emb(_near_dup(ps, noise=0.001)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "pricing/starter-v1.md"},
        },
        # PS2: NEW price (Jun 2023) — contradicts PS1
        {
            "id": "pricing-starter-002",
            "text": (
                "Starter plan is now $39/month (billed monthly) or $31/month (billed annually). "
                "Includes up to 5 users and 10GB storage."
            ),
            "embedding": emb(_near_dup(ps, noise=0.022)),
            "metadata": {"created_at": "2023-06-01T00:00:00Z", "source": "pricing/starter-v2.md"},
        },
        # PS3: feature description (no price, stable)
        {
            "id": "pricing-starter-003",
            "text": (
                "Starter plan includes: email support, 5 team members, 10GB storage, "
                "API access (100 req/min), and standard integrations."
            ),
            "embedding": emb(_cluster_member(ps, noise=0.20)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "pricing/features.md"},
        },
        {
            "id": "pricing-starter-004",
            "text": "Upgrade from Starter to Pro at any time. Prorated credit applied automatically.",
            "embedding": emb(_cluster_member(ps, noise=0.22)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "pricing/upgrades.md"},
        },
        {
            "id": "pricing-starter-005",
            "text": "Starter plan is best for small teams and individual creators getting started.",
            "embedding": emb(_cluster_member(ps, noise=0.24)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "pricing/overview.md"},
        },
        # --- Pro plan: price evolution ---
        # PP1: OLD price
        {
            "id": "pricing-pro-001",
            "text": (
                "Pro plan: $99/month (billed monthly) or $79/month (billed annually). "
                "Includes 25 users, 100GB storage, and phone support."
            ),
            "embedding": emb(_near_dup(pp, noise=0.001)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "pricing/pro-v1.md"},
        },
        # PP2: NEW price — contradicts PP1
        {
            "id": "pricing-pro-002",
            "text": (
                "Pro plan updated pricing: $119/month (billed monthly) or $95/month (billed annually). "
                "Includes 25 users, 100GB storage, and priority phone support."
            ),
            "embedding": emb(_near_dup(pp, noise=0.021)),
            "metadata": {"created_at": "2023-07-01T00:00:00Z", "source": "pricing/pro-v2.md"},
        },
        {
            "id": "pricing-pro-003",
            "text": "Pro plan users receive priority support with a guaranteed 4-hour response time.",
            "embedding": emb(_cluster_member(pp, noise=0.19)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "pricing/pro-features.md"},
        },
        # --- Enterprise plan: price contradiction ---
        {
            "id": "pricing-enterprise-001",
            "text": (
                "Enterprise plans start at $299/month for organizations with up to 100 users. "
                "Custom pricing available for larger teams."
            ),
            "embedding": emb(_near_dup(pe, noise=0.001)),
            "metadata": {
                "created_at": "2023-01-01T00:00:00Z",
                "source": "pricing/enterprise-v1.md",
            },
        },
        {
            "id": "pricing-enterprise-002",
            "text": (
                "Enterprise pricing starts at $499/month for organizations with 100 or more users. "
                "Includes dedicated support, SSO, and custom SLAs."
            ),
            "embedding": emb(_near_dup(pe, noise=0.020)),
            "metadata": {
                "created_at": "2023-09-01T00:00:00Z",
                "source": "pricing/enterprise-v2.md",
            },
        },
        {
            "id": "pricing-enterprise-003",
            "text": "Contact sales@example.com for a custom enterprise quote and demo.",
            "embedding": emb(_cluster_member(pe, noise=0.21)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "pricing/enterprise.md"},
        },
        {
            "id": "pricing-enterprise-004",
            "text": "Enterprise plan includes a 99.9% uptime SLA with financial penalties for breaches.",
            "embedding": emb(_cluster_member(pe, noise=0.20)),
            "metadata": {
                "created_at": "2023-01-01T00:00:00Z",
                "source": "pricing/enterprise-sla.md",
            },
        },
        {
            "id": "pricing-enterprise-005",
            "text": "Volume discounts start at 10% for 500+ seats and 20% for 1000+ seats.",
            "embedding": emb(_cluster_member(pe, noise=0.22)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "pricing/volume.md"},
        },
        # --- Misc pricing (stable) ---
        {
            "id": "pricing-misc-001",
            "text": "Annual billing saves up to 20% compared to month-to-month pricing.",
            "embedding": emb(_cluster_member(pm, noise=0.15)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "pricing/billing.md"},
        },
        {
            "id": "pricing-misc-002",
            "text": "All plans include a 14-day free trial. No credit card required to start.",
            "embedding": emb(_cluster_member(pm, noise=0.16)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "pricing/trial.md"},
        },
    ]


def api_collection() -> list[dict]:
    ai = B["api_install"]
    aa = B["api_auth"]
    am = B["api_methods"]

    # Pre-compute sub-bases so paired chunks share a common ancestor
    ai_python = _cluster_member(ai, noise=0.10)  # Python SDK sub-cluster
    aa_ratelimit = _cluster_member(aa, noise=0.12)  # Rate-limit sub-cluster

    return [
        # --- SDK Installation: version supersession ---
        # JS SDK v1 (old)
        {
            "id": "api-install-001",
            "text": "Install the JavaScript SDK: npm install @acme/sdk@1.2.3",
            "embedding": emb(_near_dup(ai, noise=0.001)),
            "metadata": {
                "created_at": "2023-01-15T00:00:00Z",
                "source": "docs/sdk/js-install-v1.md",
            },
        },
        # JS SDK v2 (new — supersedes api-install-001)
        {
            "id": "api-install-002",
            "text": (
                "Install the JavaScript SDK v2: npm install @acme/sdk@2.0.0\n"
                "Note: v2 contains breaking changes. See the migration guide."
            ),
            "embedding": emb(_near_dup(ai, noise=0.023)),
            "metadata": {
                "created_at": "2023-08-01T00:00:00Z",
                "source": "docs/sdk/js-install-v2.md",
            },
        },
        # Python SDK v1 (old) — shares ai_python sub-base with v2
        {
            "id": "api-install-003",
            "text": "Install the Python SDK: pip install acme-sdk==1.4.0",
            "embedding": emb(_near_dup(ai_python, noise=0.001)),
            "metadata": {
                "created_at": "2023-01-15T00:00:00Z",
                "source": "docs/sdk/python-install-v1.md",
            },
        },
        # Python SDK v2 (new — supersedes api-install-003)
        {
            "id": "api-install-004",
            "text": (
                "Install the Python SDK v2: pip install acme-sdk==2.0.0\n"
                "Python 3.9+ required. The v1 package (acme-sdk<2.0) reaches end-of-life December 2024."
            ),
            "embedding": emb(_near_dup(ai_python, noise=0.025)),
            "metadata": {
                "created_at": "2023-08-01T00:00:00Z",
                "source": "docs/sdk/python-install-v2.md",
            },
        },
        # --- Authentication (stable, with one numeric contradiction) ---
        {
            "id": "api-auth-001",
            "text": (
                "Authenticate all API requests by including your API key in the Authorization header: "
                "Authorization: Bearer <your-api-key>"
            ),
            "embedding": emb(_near_dup(aa, noise=0.001)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "docs/api/auth.md"},
        },
        {
            "id": "api-auth-002",
            "text": "Generate API keys from your dashboard: Settings → API → Create Key. Keys never expire.",
            "embedding": emb(_cluster_member(aa, noise=0.18)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "docs/api/auth.md"},
        },
        # Rate limit contradiction — both share aa_ratelimit sub-base → cosine > 0.90
        {
            "id": "api-auth-003",
            "text": (
                "API rate limit: 100 requests per minute per API key. "
                "Exceeding the limit returns HTTP 429 Too Many Requests."
            ),
            "embedding": emb(_near_dup(aa_ratelimit, noise=0.001)),
            "metadata": {"created_at": "2023-01-01T00:00:00Z", "source": "docs/api/rate-limits.md"},
        },
        # Contradicts api-auth-003 (different rate limit) — near-dup of same sub-base
        {
            "id": "api-auth-004",
            "text": (
                "Pro and Enterprise plans support up to 1000 requests per minute per API key. "
                "Starter plan is limited to 100 requests per minute."
            ),
            "embedding": emb(_near_dup(aa_ratelimit, noise=0.025)),
            "metadata": {
                "created_at": "2023-06-01T00:00:00Z",
                "source": "docs/api/rate-limits-v2.md",
            },
        },
        # --- Core methods: v1 → v2 supersession ---
        # v1 initialization (old)
        {
            "id": "api-methods-001",
            "text": (
                "Initialize the client:\n"
                "const client = new AcmeClient({ apiKey: 'your-key' });\n"
                "await client.connect();"
            ),
            "embedding": emb(_near_dup(am, noise=0.001)),
            "metadata": {
                "created_at": "2023-01-15T00:00:00Z",
                "source": "docs/sdk/quickstart-v1.md",
            },
        },
        # v2 initialization (supersedes api-methods-001)
        {
            "id": "api-methods-002",
            "text": (
                "In v2.0, initialize the client using initialize() instead of connect():\n"
                "const client = new AcmeClient({ apiKey: 'your-key' });\n"
                "await client.initialize();\n"
                "The connect() method is deprecated and will be removed in v3.0."
            ),
            "embedding": emb(_near_dup(am, noise=0.025)),
            "metadata": {
                "created_at": "2023-08-01T00:00:00Z",
                "source": "docs/sdk/quickstart-v2.md",
            },
        },
        # Stable methods
        {
            "id": "api-methods-003",
            "text": "Query records: const results = await client.query({ filter: { status: 'active' } });",
            "embedding": emb(_cluster_member(am, noise=0.19)),
            "metadata": {"created_at": "2023-01-15T00:00:00Z", "source": "docs/sdk/querying.md"},
        },
        {
            "id": "api-methods-004",
            "text": ("Register webhooks: await client.webhooks.register(url, ['record.created', 'record.updated'])"),
            "embedding": emb(_cluster_member(am, noise=0.20)),
            "metadata": {"created_at": "2023-01-15T00:00:00Z", "source": "docs/sdk/webhooks.md"},
        },
        {
            "id": "api-methods-005",
            "text": "All SDK methods return Promises. Use async/await or .then()/.catch() for error handling.",
            "embedding": emb(_cluster_member(am, noise=0.21)),
            "metadata": {
                "created_at": "2023-01-15T00:00:00Z",
                "source": "docs/sdk/error-handling.md",
            },
        },
        {
            "id": "api-methods-006",
            "text": "The SDK ships full TypeScript definitions. No @types package needed.",
            "embedding": emb(_cluster_member(am, noise=0.22)),
            "metadata": {"created_at": "2023-01-15T00:00:00Z", "source": "docs/sdk/typescript.md"},
        },
        {
            "id": "api-methods-007",
            "text": (
                "Batch operations: process up to 100 records in a single call using client.batch(ops). "
                "Reduces API round-trips and counts as a single request against your rate limit."
            ),
            "embedding": emb(_cluster_member(am, noise=0.23)),
            "metadata": {"created_at": "2023-01-15T00:00:00Z", "source": "docs/sdk/batch.md"},
        },
    ]


# ---------------------------------------------------------------------------
# Write collections
# ---------------------------------------------------------------------------


def write_collection(name: str, chunks: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(chunks, indent=2))
    print(f"  {path}  ({len(chunks)} chunks)")


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "corpus"
    print(f"Generating benchmark corpus → {out_dir}")

    docs = docs_collection()
    pricing = pricing_collection()
    api = api_collection()
    total = len(docs) + len(pricing) + len(api)

    write_collection("docs", docs, out_dir)
    write_collection("pricing", pricing, out_dir)
    write_collection("api", api, out_dir)

    print(f"\nTotal: {total} chunks across 3 collections")
    print("\nDesigned signals:")
    print("  Near-duplicates (lexical)  : docs-refund-001/002, docs-support-001/002")
    print("  Near-duplicates (semantic) : docs-refund-001/003")
    print("  Numeric contradictions     : pricing-starter-001 vs 002, pro-001 vs 002, enterprise-001 vs 002")
    print("  Version supersession       : api-install-001→002 (JS), 003→004 (Python)")
    print("                             : api-methods-001→002 (connect→initialize)")
    print("  Rate-limit contradiction   : api-auth-003 vs api-auth-004")
    print("  Time-bounded content       : docs-promo-001/002/003")
    print("  Orphan chunks              : docs-orphan-001 through 006")
    print("\nEmbedding dimension: 384 (sentence-transformers/all-MiniLM-L6-v2 compatible)")
