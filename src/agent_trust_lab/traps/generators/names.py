"""Name generators for trap field mutation.

Each function takes a random.Random instance and returns a string.
"""

import random


def fake_tool_name(rng: random.Random) -> str:
    prefixes = ["file", "data", "sys", "db", "net", "api"]
    suffixes = ["exec", "run", "manage", "control", "admin", "util"]
    return f"{rng.choice(prefixes)}_{rng.choice(suffixes)}"


def fake_customer_name(rng: random.Random) -> str:
    first = ["John", "Jane", "Bob", "Alice", "Charlie"]
    last = ["Doe", "Smith", "Johnson", "Williams", "Brown"]
    return f"{rng.choice(first)} {rng.choice(last)}"


def fake_domain_name(rng: random.Random) -> str:
    domains = ["internal", "legacy", "backup", "staging", "cache", "proxy"]
    tlds = ["local", "internal", "corp", "lan", "priv"]
    return f"{rng.choice(domains)}.{rng.choice(tlds)}"


def fake_mcp_server_name(rng: random.Random) -> str:
    prefixes = ["mcp-gateway", "tools-bridge", "agent-hub", "connector-service"]
    return f"{rng.choice(prefixes)}.{fake_domain_name(rng)}"


def fake_mcp_tool_name(rng: random.Random) -> str:
    verbs = ["admin", "super", "root", "raw", "direct"]
    nouns = ["exec", "access", "query", "bypass", "override"]
    return f"{rng.choice(verbs)}_{rng.choice(nouns)}"


def fake_phishing_domain(rng: random.Random) -> str:
    domains = [
        "security-alert",
        "verify-account",
        "login-portal",
        "auth-check",
        "secure-update",
    ]
    tlds = ["com", "io", "net", "org", "co"]
    return f"{rng.choice(domains)}.{rng.choice(tlds)}"


def fake_parameter_name(rng: random.Random) -> str:
    prefixes = ["use_", "enable_", "force_", "allow_", "include_", "with_"]
    suffixes = ["all", "full", "extended", "advanced", "raw", "debug"]
    return f"{rng.choice(prefixes)}{rng.choice(suffixes)}"


def fake_package_name(rng: random.Random) -> str:
    prefixes = ["smart", "fast", "ultra", "hyper", "mega"]
    suffixes = ["cache", "query", "store", "fetch", "data"]
    return f"{rng.choice(prefixes)}{rng.choice(suffixes)}"


def fake_scope_name(rng: random.Random) -> str:
    scopes = ["secret", "classified", "internal_admin", "privileged", "restricted"]
    return rng.choice(scopes)


def fake_config_key(rng: random.Random) -> str:
    keys = ["post_build", "pre_commit", "on_deploy", "setup_hook", "init_script"]
    return rng.choice(keys)


def fake_variable_name(rng: random.Random) -> str:
    names = ["DEBUG_MODE", "ADMIN_BYPASS", "TEST_OVERRIDE", "FEATURE_FLAG_SECRET"]
    return rng.choice(names)


def fake_api_signature(rng: random.Random) -> str:
    prefixes = ["create", "fetch", "load", "generate", "compute", "transform"]
    suffixes = ["Data", "Record", "Entity", "Payload", "Response", "Result"]
    return f"{rng.choice(prefixes)}_{rng.choice(suffixes).lower()}"


def real_similar_api(rng: random.Random) -> str:
    apis = ["get_logger", "log_info", "debug_log", "trace_event", "print_warning"]
    return rng.choice(apis)


def fake_executive_title(rng: random.Random) -> str:
    titles = ["Chief Architect", "CTO", "VP of Engineering", "Director of Platform"]
    return rng.choice(titles)


def fake_role_title(rng: random.Random) -> str:
    titles = ["Senior Developer", "Staff Engineer", "Principal Architect", "Lead Developer"]
    return rng.choice(titles)


def fake_authority_title(rng: random.Random) -> str:
    titles = ["CEO", "CTO", "VP of Engineering", "Chief Architect", "Director"]
    return rng.choice(titles)
