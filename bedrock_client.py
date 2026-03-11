"""
AWS Bedrock client for Claude Sonnet.
Uses the WMO-AWS-Toolbox-Console credential_process configured in ~/.aws/config.
"""

import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound

# ── Configuration ──────────────────────────────────────────────────────────────
AWS_PROFILE  = "aws-cus-common"          # name of the profile in ~/.aws/config
AWS_REGION   = "eu-west-1"              # change to us-east-1 if Bedrock not in eu-west-1
MODEL_ID     = "eu.anthropic.claude-sonnet-4-6"
MAX_TOKENS   = 4096
# ───────────────────────────────────────────────────────────────────────────────


def get_bedrock_client():
    """Return a boto3 bedrock-runtime client using the configured AWS profile."""
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client("bedrock-runtime")


def invoke_claude(prompt: str, system: str | None = None) -> str:
    """
    Send a prompt to Claude Sonnet via Amazon Bedrock and return the text response.

    Args:
        prompt:  The user message to send.
        system:  Optional system-level instruction.

    Returns:
        The model's text reply as a string.
    """
    client = get_bedrock_client()

    messages = [{"role": "user", "content": prompt}]

    body: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "messages": messages,
    }
    if system:
        body["system"] = system

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    except ProfileNotFound:
        raise RuntimeError(
            f"AWS profile '{AWS_PROFILE}' not found in ~/.aws/config.\n"
            "See the README section 'AWS Profile Setup' for instructions."
        )
    except NoCredentialsError:
        raise RuntimeError(
            "No valid AWS credentials found. "
            "Run: aws sts get-caller-identity --profile aws-cus-common  to test auth."
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = e.response["Error"]["Message"]
        raise RuntimeError(f"Bedrock API error [{code}]: {msg}") from e
