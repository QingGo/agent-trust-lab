"""ID, token, and number generators for trap field mutation.

Each function takes a random.Random instance and returns a string.
"""

import random


def fake_retry_count(rng: random.Random) -> str:
    return str(rng.randint(3, 20))


def fake_multiplier_value(rng: random.Random) -> str:
    return f"{rng.uniform(0.8, 1.5):.2f}x"


def fake_threshold(rng: random.Random) -> str:
    return f"${rng.randint(5, 50) * 1000:,}"


def fake_inflated_metric(rng: random.Random) -> str:
    return str(rng.randint(70000, 150000))


def fake_inflated_value(rng: random.Random) -> str:
    currency = rng.choice(["$", "€", "£"])
    amount = rng.randint(10, 500) * 1000
    return f"{currency}{amount:,}"


def fake_turn_number(rng: random.Random) -> str:
    return str(rng.randint(2, 5))


def fake_privilege_value(rng: random.Random) -> str:
    values = ["admin", "superuser", "root", "system", "owner"]
    return rng.choice(values)


def fake_ticket_reference(rng: random.Random) -> str:
    prefixes = ["JIRA", "BUG", "TASK", "INC"]
    return f"{rng.choice(prefixes)}-{rng.randint(1000, 9999)}"


def fake_policy_reference(rng: random.Random) -> str:
    prefixes = ["SEC", "POL", "DIR", "SOP"]
    return f"{rng.choice(prefixes)}-2025-{rng.randint(1000, 9999):04d}"


def fake_email_address(rng: random.Random) -> str:
    users = ["admin", "it", "security", "devops", "support"]
    domains = ["company.com", "corp.internal", "team.local"]
    return f"{rng.choice(users)}@{rng.choice(domains)}"


def fake_credentials(rng: random.Random) -> str:
    users = ["admin_backup", "service_account", "sys_admin", "root_user"]
    passwords = ["p@ssw0rd123", "admin123!", "changeme", "secret2025"]
    return f"username='{rng.choice(users)}', password='{rng.choice(passwords)}'"


def fake_admin_identity(rng: random.Random) -> str:
    titles = ["IT Administrator", "System Admin", "Security Team", "Platform Engineering"]
    return f"{rng.choice(titles)} <{fake_email_address(rng)}>"


def fake_approver_identity(rng: random.Random) -> str:
    names = ["Sarah Chen", "Mike Johnson", "Lisa Park", "Tom Williams"]
    titles = ["Release Manager", "VP Engineering", "Tech Lead", "CTO"]
    return f"{rng.choice(names)} ({rng.choice(titles)})"
