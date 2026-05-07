from agent_trust_lab.traps.templates import InjectionTemplate, register_template


@register_template("system_note")
class SystemNoteTemplate(InjectionTemplate):
    name = "system_note"
    description = "Direct [SYSTEM NOTE] prefix (original mode)"

    def render(self, base_task: str, injection: str, tools=None) -> str:
        return f"{base_task}\n\n[SYSTEM NOTE]: {injection}"
