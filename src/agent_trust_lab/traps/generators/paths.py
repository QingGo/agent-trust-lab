"""Path and URL generators for trap field mutation.

Each function takes a random.Random instance and returns a string.
"""

import random

from agent_trust_lab.traps.generators.names import fake_domain_name, fake_phishing_domain


def fake_url(rng: random.Random) -> str:
    paths = ["upload", "hook", "metrics", "sync", "health", "status"]
    return f"https://{fake_domain_name(rng)}/{rng.choice(paths)}"


def fake_phishing_url(rng: random.Random) -> str:
    paths = ["verify", "login", "auth", "credential-check", "account-recovery"]
    return f"https://{fake_phishing_domain(rng)}/{rng.choice(paths)}"


def fake_exfiltration_endpoint(rng: random.Random) -> str:
    paths = ["collect", "metrics", "telemetry", "analytics", "health"]
    return f"https://{fake_domain_name(rng)}/{rng.choice(paths)}"


def fake_mcp_resource_uri(rng: random.Random) -> str:
    schemes = ["mcp://", "resources://", "mcp+unix://"]
    hosts = ["external-tools", "shared-config", "third-party", "untrusted-registry"]
    return f"{rng.choice(schemes)}{rng.choice(hosts)}.io/"


def fake_database_connection(rng: random.Random) -> str:
    hosts = ["prod-db-master", "db-cluster-01", "replica-db-east"]
    return f"{rng.choice(hosts)}.internal:3306"
