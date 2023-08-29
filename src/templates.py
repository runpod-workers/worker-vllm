class Template():
    def __init__(self, template_method):
        self.template_method = template_method

    def __call__(self, prompt):
        return self.template_method(prompt)


LLAMA2_TEMPLATE = Template(
    lambda prompt: """SYSTEM: You are a helpful assistant.
USER: {}
ASSISTANT: """.format(prompt)
)

DEFAULT_TEMPLATE = Template(
    lambda prompt: prompt
)
