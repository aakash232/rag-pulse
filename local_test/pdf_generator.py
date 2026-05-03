import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_DIR = "pdfs"
PAGES_PER_DOC = 60
LINES_PER_PAGE = 40

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_section(prompt):
    """Cheap OpenAI call for structured content"""
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "Write structured technical documentation."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content


def draw_page(c, lines):
    y = 750
    for line in lines:
        c.drawString(40, y, line[:95])
        y -= 15


def expand_to_pages(base_text):
    """Repeat + slightly vary text to fill pages"""
    lines = []
    base_lines = base_text.split("\n")

    while len(lines) < PAGES_PER_DOC * LINES_PER_PAGE:
        lines.extend(base_lines)

    return lines[: PAGES_PER_DOC * LINES_PER_PAGE]


def write_pdf(filename, content_lines):
    c = canvas.Canvas(filename, pagesize=letter)

    for i in range(PAGES_PER_DOC):
        page_lines = content_lines[i * LINES_PER_PAGE : (i + 1) * LINES_PER_PAGE]
        draw_page(c, page_lines)
        c.showPage()

    c.save()


# -----------------------------
# DOCUMENT GENERATORS
# -----------------------------


def api_guidelines(version):
    return generate_section(f"""
    Write API design guidelines version {version}.
    Include:
    - authentication approach
    - rate limits
    - error handling
    Keep it consistent and realistic.
    """)


def security_policy(version):
    if version == 1:
        rule = "OAuth 2.0 is mandatory. API keys are forbidden."
    else:
        rule = "API keys are standard. OAuth is optional."

    return generate_section(f"""
    Write a security policy document.
    Core rule: {rule}
    Include encryption, auth, and rate limiting sections.
    """)


def deployment_guide(year):
    if year == 2019:
        stack = "Docker Swarm, Ubuntu 18.04, Jenkins"
    else:
        stack = "Kubernetes, Ubuntu 22.04, GitHub Actions, ArgoCD"

    return generate_section(f"""
    Write a deployment guide for year {year}.
    Stack: {stack}
    Include CI/CD and monitoring.
    """)


def architecture(version):
    return generate_section(f"""
    Write system architecture version {version}.
    Include:
    - components
    - data flow
    - scaling strategy
    """)


def postmortem():
    return generate_section("""
    Write an incident postmortem with:
    - timeline
    - root cause
    - mitigation
    Slightly messy but still readable.
    """)


def product_spec():
    return generate_section("""
    Write a product requirements document.
    Include:
    - duplicated requirements
    - one conflicting requirement
    - some noisy notes
    """)


# -----------------------------
# BUILD ALL PDFs
# -----------------------------


def build_all():
    docs = [
        ("api_v1.pdf", api_guidelines(1)),
        ("api_v2.pdf", api_guidelines(2)),
        ("security_v1.pdf", security_policy(1)),
        ("security_v2.pdf", security_policy(2)),
        ("deploy_2019.pdf", deployment_guide(2019)),
        ("deploy_2025.pdf", deployment_guide(2025)),
        ("arch_v1.pdf", architecture(1)),
        ("arch_v2.pdf", architecture(2)),
        ("postmortem.pdf", postmortem()),
        ("product_spec.pdf", product_spec()),
    ]

    for filename, base_text in docs:
        print(f"Generating {filename}...")

        expanded = expand_to_pages(base_text)
        full_path = os.path.join(OUTPUT_DIR, filename)

        write_pdf(full_path, expanded)

    print("✅ All PDFs generated in /pdfs")


if __name__ == "__main__":
    build_all()
