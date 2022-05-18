## NOTE: This script assumes you've run the secondary labels script beforehand
import pandas as pd

# Get only the plaintiffs labelled as Medical
options = ['MEDICAL', 'MEDICAL_EMBED_GLOVE', 'MEDICAL_EMBED_W2V', 'MEDICAL_EMBED_BOTH']
is_medical_embed = plaintiffs_clean["Entity"].isin(options)
medical_embed = plaintiffs_clean[is_medical_embed]

# Extract the debt amounts (found in misc column) into its own column
debt_amounts = []

for index, row in medical_embed.iterrows():
    misc_json = row['misc']
    if "Complaint Information" in misc_json:
        amount = misc_json['Complaint Information']['Amount']
        if isinstance(amount, list):
            amount = amount[0].strip("$")
        else:
            amount = amount.strip("$")
                
    else:
        amount = math.nan

    debt_amounts.append(amount)

medical_embed.insert(2, "Debt Amounts", debt_amounts, True)

# Extract the plaintiffs with non-zero debt amounts into their own dataframe
has_debt_amount = pd.notna(medical_embed["Debt Amounts"])
debt_amounts_frame = medical_embed[has_debt_amount]
debt_amounts_frame["Debt Amounts"]

# Use this new frame to do category-based debt lookups
options = ['AMBULANCE']
is_ambulance = plaintiffs_clean["Secondary Label"].isin(options)
ambulance_debt = debt_amounts_frame[is_ambulance]
ambulance_debt

options = ['HOSPITAL']
is_hospital = plaintiffs_clean["Secondary Label"].isin(options)
hospital_debt = debt_amounts_frame[is_hospital]
hospital_debt

options = ['NURSE', 'NURSING']
is_hospital = plaintiffs_clean["Secondary Label"].isin(options)
hospital_debt = debt_amounts_frame[is_hospital]
hospital_debt