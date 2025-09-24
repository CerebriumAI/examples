#!/usr/bin/env python3
"""
Helper script to generate PayPal access tokens using Client Credentials flow.
This is for server-side authentication without browser interaction.
"""

import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()


def get_paypal_access_token(
    client_id: str, client_secret: str, environment: str = "sandbox"
):
    """
    Generate PayPal access token using Client Credentials flow.

    Args:
        client_id: PayPal app client ID
        client_secret: PayPal app client secret
        environment: "sandbox" or "live"

    Returns:
        str: Access token
    """
    if environment == "sandbox":
        base_url = "https://api-m.sandbox.paypal.com"
    else:
        base_url = "https://api-m.paypal.com"

    # Encode client credentials
    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        "Accept": "application/json",
        "Accept-Language": "en_US",
        "Authorization": f"Basic {encoded_credentials}",
    }

    data = {
        "grant_type": "client_credentials",
        "response_type": "token",
        "intent": "sdk_init",
    }

    response = requests.post(f"{base_url}/v1/oauth2/token", headers=headers, data=data)

    if response.status_code == 200:
        token_data = response.json()
        return token_data["access_token"]
    else:
        raise Exception(
            f"Failed to get access token: {response.status_code} - {response.text}"
        )


if __name__ == "__main__":
    client_id = os.getenv("PAYPAL_CLIENT_ID")
    client_secret = os.getenv("PAYPAL_CLIENT_SECRET")
    environment = os.getenv("PAYPAL_ENVIRONMENT", "sandbox")

    if not client_id or not client_secret:
        print(
            "Please set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET environment variables"
        )
        exit(1)

    try:
        access_token = get_paypal_access_token(client_id, client_secret, environment)
        print(f"Access Token: {access_token}")

        # Save to .env file
        with open(".env", "a") as f:
            f.write(f"\nPAYPAL_ACCESS_TOKEN={access_token}\n")
        print("Access token saved to .env file")

    except Exception as e:
        print(f"Error: {e}")
