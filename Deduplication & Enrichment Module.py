import pandas as pd
from fuzzywuzzy import fuzz, process
import openai

openai.api_key = "YOUR_API_KEY"

# Demo data: replace with your scraped lead CSV
df = pd.DataFrame([
    {"Name": "Jane Doe", "Email": "jane@acme.com", "Company": "Acme Corp", "LinkedIn": "", "Job_Title": ""},
    {"Name": "Jane D.", "Email": "jane@acme.com", "Company": "", "LinkedIn": "linkedin.com/in/janedoe", "Job_Title": "Head of Growth"},
    {"Name": "John Smith", "Email": "john.smith@xyz.com", "Company": "XYZ Inc", "LinkedIn": "linkedin.com/in/jsmith", "Job_Title": None},
    {"Name": "Jon Smith", "Email": "john.smith@xyz.com", "Company": "", "LinkedIn": "linkedin.com/in/johnsmith", "Job_Title": "Sales Director"},
])

def fuzzy_dedupe(df):
    deduped = []
    seen_emails = {}
    for idx, row in df.iterrows():
        email = row['Email'].lower()
        if email in seen_emails:
            # Merge logic: combine non-empty fields
            prev_idx = seen_emails[email]
            for col in df.columns:
                if not pd.isnull(row[col]) and row[col] != "":
                    df.at[prev_idx, col] = row[col]
        else:
            seen_emails[email] = idx
            deduped.append(idx)
    return df.loc[deduped].reset_index(drop=True)

def enrich_lead(row):
    # Fills missing Job_Title/Company/LinkedIn using GPT
    fields = []
    if not row['Job_Title']:
        fields.append("job title")
    if not row['Company']:
        fields.append("company")
    if not row['LinkedIn']:
        fields.append("LinkedIn profile")

    # Build prompt and enrich via OpenAI only if needed
    if fields:
        prompt = f"Given the person named {row['Name']} with this email {row['Email']}, can you guess their likely {', and '.join(fields)} in a business context? Return JSON {{'Job_Title':'', 'Company':'', 'LinkedIn':''}}"
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"user","content":prompt}]
            )
            content = completion.choices[0].message['content']
            ai_result = eval(content) if '{' in content else {}  # Caution: Use json.loads in practice!
            for key in ['Job_Title', 'Company', 'LinkedIn']:
                if key in ai_result and ai_result[key] and not row.get(key):
                    row[key] = ai_result[key]
        except Exception as ex:
            print("OpenAI error, skipping enrichment:", ex)
    return row

# Step 1. Deduplicate
df_deduped = fuzzy_dedupe(df)

# Step 2. Enrich missing fields
df_enriched = df_deduped.apply(enrich_lead, axis=1)

print("Enriched & Deduplicated Leads:\n", df_enriched)
