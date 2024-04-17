import pandas as pd

df = pd.read_csv('filtered_values.csv')


# Function to generate a descriptive text for each row
def create_description(row):
    
    # Sex, YoB, Age
    description = f"You are a {row['Sex']} born in {row['Year_of_Birth']}, which means that you are {row['Age']} years old. "

    if row['Is_Immigrant'] == 'yes':
        description += f"You were born in {row['Country_of_Birth']} and you live in {row['Country_of_Residence']}, which means that you are an immigrant. "
    else:
        description += f"You were born in {row['Country_of_Birth']} and you live in {row['Country_of_Residence']}, which means that you are not an immigrant. "

    if row['Is_Immigrant_Mother'] == 'yes':
        description += f"Your Mother is from {row['Country_of_Birth_Mother']}, and she is an immigrant. "
    else:
        description += f"Your Mother is from {row['Country_of_Birth_Mother']}, and she is not an immigrant. "
    
    if row['Is_Immigrant_Father'] == 'yes':
        description += f"Your Father is from {row['Country_of_Birth_Father']}, and he is an immigrant. "
    else:
        description += f"Your Father is from {row['Country_of_Birth_Father']}, and he is not an immigrant. "
    
    description += f"{row['n_People_in_Household']} people live in your household, and "

    if row['Live_with_Parents'] == "no":
        description += "you don't live with your parents. "
    else:
        description += f"you live with {row['Live_with_Parents']}. "

    description += f"You are {row['Marital_Status']}, and have {row['n_Children']} children. "

    description += f"You have received {row['Education']}, your spouse {row['Education_Spouse']}, your mother {row['Education_Mother']}, and your father {row['Education_Father']}. "

    description += f"You are {row['Employment_Status']}, and your spouse is {row['Employment_Status_Spouse']}. "

    description += f"You work in {row['Occupational_Group']}, your spouse in {row['Occupational_Group_Spouse']}, and your father in {row['Occupational_Group_Father']}. "

    description += f"You work for {row['Works_for']}. "

    if row['Chief_Wage_Earner'] == 'yes':
        description += f"You are the chief wage earner of the household, and during the past year, your household {row['Last_Year_Savings']}. "
    else:
        description += f"You are not the chief wage earner of the household, and during the past year, your household {row['Last_Year_Savings']}. "
    
    description += f"You consider yourself a {row['Self_Assessed_Social_Class']} and your income level is {row['Income_Level']} out of 10, where 10 is the highest and 1 is the lowest. "

    if row['Religious_Denomination'] == 'do not belong to a religious denomination':
        description += f"Lastly, you do not belong to a religious denomination."
    else:
        description += f"Lastly, you are a {row['Religious_Denomination']}."

    return description
# Apply the function to each row
df['Description'] = df.apply(create_description, axis=1)

# Save the DataFrame to a new CSV file, "enhanced_data.csv"
# In a local environment, use: df.to_csv("enhanced_data.csv", index=False)
df.to_csv('Values_processed_desc.csv', index=False)