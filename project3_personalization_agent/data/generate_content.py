"""Generate the synthetic content catalog and user profiles."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

OUT_DIR = Path(__file__).parent
ITEMS_PATH = OUT_DIR / "content_items.json"
USERS_PATH = OUT_DIR / "user_profiles.json"

TOPICS = [
    "sms-marketing",
    "email-deliverability",
    "lifecycle-automation",
    "loyalty-programs",
    "audience-segmentation",
    "ai-personalization",
    "post-purchase-flows",
    "winback-campaigns",
    "mobile-commerce",
    "consent-and-privacy",
    "rcs-messaging",
    "creative-testing",
    "attribution",
    "subscription-retention",
    "first-party-data",
]

FUNNEL_STAGES = ["awareness", "consideration", "decision", "post_purchase"]

PERSONAS = [
    "lifecycle_marketer",
    "growth_lead",
    "marketing_ops",
    "ecommerce_director",
    "agency_partner",
    "rev_ops",
]


@dataclass
class ContentItem:
    item_id: str
    title: str
    body: str
    topics: list[str]
    funnel_stage: str
    persona_fit: list[str]


@dataclass
class UserProfile:
    user_id: str
    recent_topics: list[str]
    target_stage: str
    persona: str
    relevant_item_ids: list[str]


def _stub_body(rng: random.Random, topics: list[str], stage: str) -> str:
    openers = [
        "Across hundreds of programs, the same patterns surface. ",
        "If your team is rethinking this, you're not alone. ",
        "Most teams start in one place and end up somewhere else. ",
        "There is a quiet disagreement underneath this question. ",
        "The fastest-moving teams handle this differently. ",
    ]
    middles = [
        f"In the {stage} stage, audiences expect very different signals. ",
        f"Topics like {', '.join(topics[:2])} drive most of the variance. ",
        "Calibration matters more than novelty here. ",
        "What looks like a tactic question is usually a system question. ",
    ]
    closers = [
        "We unpack the trade-offs and the reasoning behind each choice.",
        "This piece walks through the framework and the data behind it.",
        "Inside: a checklist your team can adapt this quarter.",
        "We share what worked, what stalled, and what we'd do differently.",
    ]
    return rng.choice(openers) + rng.choice(middles) + rng.choice(closers)


def _make_title(rng: random.Random, topics: list[str], stage: str) -> str:
    primary = topics[0].replace("-", " ")
    by_stage = {
        "awareness": [
            f"Why {primary} is changing in 2026",
            f"The state of {primary}: a primer",
            f"Three things teams misread about {primary}",
        ],
        "consideration": [
            f"How leading teams approach {primary}",
            f"Compare: traditional vs agent-driven {primary}",
            f"A walkthrough of modern {primary} stacks",
        ],
        "decision": [
            f"Pricing and rollout for our {primary} platform",
            f"Book a demo focused on {primary}",
            f"Migration guide: switching to AI-augmented {primary}",
        ],
        "post_purchase": [
            f"Onboarding playbook for {primary}",
            f"Power-user tips for {primary}",
            f"Renewal-ready: optimizing your {primary} program",
        ],
    }
    return rng.choice(by_stage[stage])


def generate_items(n: int = 300, seed: int = 13) -> list[ContentItem]:
    rng = random.Random(seed)
    items: list[ContentItem] = []
    for i in range(n):
        topic_count = rng.choice([1, 2, 2, 3])
        topics = rng.sample(TOPICS, topic_count)
        stage = rng.choice(FUNNEL_STAGES)
        persona_count = rng.choice([1, 2, 2, 3])
        persona_fit = rng.sample(PERSONAS, persona_count)

        title = _make_title(rng, topics, stage)
        body = _stub_body(rng, topics, stage)
        items.append(
            ContentItem(
                item_id=f"item_{i:04d}",
                title=title,
                body=body,
                topics=topics,
                funnel_stage=stage,
                persona_fit=persona_fit,
            )
        )
    return items


def generate_users(items: list[ContentItem], n: int = 50, seed: int = 19) -> list[UserProfile]:
    """Each user's relevant_item_ids = items matching at least one recent topic AND
    matching both the target stage and the persona. Users with <3 relevant items
    are dropped so the eval signal isn't dominated by noise."""
    rng = random.Random(seed)
    users: list[UserProfile] = []
    for i in range(n):
        recent = rng.sample(TOPICS, rng.choice([2, 3, 3, 4]))
        stage = rng.choice(FUNNEL_STAGES)
        persona = rng.choice(PERSONAS)
        relevant = [
            it.item_id
            for it in items
            if (set(it.topics) & set(recent))
            and it.funnel_stage == stage
            and persona in it.persona_fit
        ]
        if len(relevant) < 3:
            continue
        users.append(
            UserProfile(
                user_id=f"user_{i:03d}",
                recent_topics=recent,
                target_stage=stage,
                persona=persona,
                relevant_item_ids=relevant[:20],
            )
        )
    return users


def write_all() -> None:
    items = generate_items()
    users = generate_users(items)
    ITEMS_PATH.write_text(json.dumps([asdict(it) for it in items], indent=2))
    USERS_PATH.write_text(json.dumps([asdict(u) for u in users], indent=2))
    print(f"Wrote {len(items)} items     -> {ITEMS_PATH}")
    print(f"Wrote {len(users)} profiles  -> {USERS_PATH}")


if __name__ == "__main__":
    write_all()
