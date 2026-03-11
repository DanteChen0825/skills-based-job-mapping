"""
Quick connectivity test — asks Claude "What is 2 + 2?" and prints the answer.
Run:  python test_connection.py
"""

from bedrock_client import invoke_claude

def main():
    print("Testing AWS Bedrock → Claude Sonnet connection …\n")
    answer = invoke_claude("What is 1 + 1? Reply with just the number.")
    print(f"Model answered: {answer}")
    print("\n✅ Connection successful!")

if __name__ == "__main__":
    main()
