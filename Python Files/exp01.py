"""
CS23334 Fundamentals of Data Science Lab - Experiment 1
"""

import pandas as pd
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet


def exp_1a_job_postings_trend():
    """
    Exp No: 1.a
    Analyze the trend of data science job postings over the last decade.
    """
    print("--- Exp 1.a: Trend of Data Science Job Postings ---")
    data = {
        'Year': list(range(2010, 2021)),
        'Job Postings': [150, 300, 450, 600, 800, 1200, 1600, 2100, 2700, 3400, 4200]
    }
    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.plot(df['Year'], df['Job Postings'], marker='o')
    plt.title('Trend of Data Science Job Postings')
    plt.xlabel('Year')
    plt.ylabel('Number of Job Postings')
    plt.tight_layout()
    plt.show()


def exp_1b_roles_distribution():
    """
    Exp No: 1.b
    Analyze and visualize the distribution of various data science roles
    (Data Analyst, Data Engineer, Data Scientist, etc.) from a dataset.
    """
    print("\n--- Exp 1.b: Distribution of Data Science Roles ---")
    roles = [
        'Data Analyst',
        'Data Engineer',
        'Data Scientist',
        'ML Engineer',
        'Business Analyst'
    ]
    counts = [300, 500, 450, 200, 150]

    for role, count in zip(roles, counts):
        print(f"{role}: {count}")

    plt.figure(figsize=(8, 5))
    plt.bar(roles, counts)
    plt.title('Distribution of Data Science Roles')
    plt.xlabel('Role')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def exp_1c_data_types():
    """
    Exp No: 1.c
    Conduct an experiment to differentiate Structured, Unstructured, and
    Semi-structured data based on datasets given.
    """
    print("\n--- Exp 1.c: Structured, Unstructured, and Semi-structured Data ---")
    # Structured data example (Tabular DataFrame)
    structured_data = pd.DataFrame({
        'ID': [1, 2, 3],
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]
    })
    print("Structured Data:\n", structured_data)

    # Unstructured data example (Free-form text)
    unstructured_data = (
        "This is an example of unstructured data. It can be a piece of text, "
        "an image, or a video file."
    )
    print("\nUnstructured Data:\n", unstructured_data)

    # Semi-structured data example (JSON / Dictionary)
    semi_structured_data = {
        'ID': 1,
        'Name': 'Alice',
        'Attributes': {'Height': 165, 'Weight': 68}
    }
    print("\nSemi-structured Data:\n", semi_structured_data)


def exp_1d_encryption_decryption():
    """
    Exp No: 1.d
    Conduct an experiment to encrypt and decrypt given sensitive data.
    """
    print("\n--- Exp 1.d: Encrypt and Decrypt Sensitive Data ---")
    # Generate key and initialize cipher suite
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)

    plain_text = b"Rajalakshmi Engineering College."
    cipher_text = cipher_suite.encrypt(plain_text)

    # Decrypt data
    decrypted_text = cipher_suite.decrypt(cipher_text)

    print("Original Data:", plain_text)
    print("Encrypted Data:", cipher_text)
    print("Decrypted Data:", decrypted_text)


if __name__ == "__main__":
    exp_1a_job_postings_trend()
    exp_1b_roles_distribution()
    exp_1c_data_types()
    exp_1d_encryption_decryption()
