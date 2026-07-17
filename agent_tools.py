import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.tools import tool
from ddgs import DDGS

@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a given location.

    Args:
        location: The city name (required), e.g. "Pune", "Mumbai", "London".
                  Do not pass "today" or "here" — ask the user for a city if unknown.

    Returns:
        A short natural-language sentence describing the current temperature, conditions, and wind.
        Returns an error message if location is not found or network issues occur.
    """
    location = (location or "").strip()
    if not location or location.lower() in {"today", "now", "here", "outside", "my location"}:
        return "Please tell me which city you want the weather for."

    try:
        # Geocode the location using Open-Meteo's free geocoding endpoint
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}"
        geo_response = requests.get(geo_url, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Sorry, I couldn't find the location: {location}"

        # Take the first result
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        # Fetch current weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,windspeed_10m&timezone=auto"
        weather_response = requests.get(weather_url, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data.get("current", {})
        temp = current.get("temperature_2m")
        windspeed = current.get("windspeed_10m")
        weather_code = current.get("weather_code")

        # Map weather code to description (simplified mapping)
        weather_descriptions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            56: "Light freezing drizzle",
            57: "Dense freezing drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Light freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow fall",
            73: "Moderate snow fall",
            75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }

        condition = weather_descriptions.get(weather_code, "Unknown")

        return f"The current temperature in {location} is {temp}°C with {condition.lower()} and wind speed of {windspeed} km/h."

    except requests.exceptions.RequestException as e:
        return f"Network error while fetching weather for {location}: {str(e)}"
    except Exception as e:
        return f"An error occurred while getting weather for {location}: {str(e)}"

@tool
def send_email(to: str, subject: str, body: str, account: str = "") -> str:
    """
    Send an email via Gmail SMTP.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body content.
        account: Optional nickname of the Gmail account to use (from JARVIS_GMAIL_ACCOUNTS).
                 If empty, uses JARVIS_GMAIL_DEFAULT or first configured account.

    Returns:
        A confirmation message on success or an error message on failure.
        Never returns or logs the app password.
    """
    try:
        # Import here to avoid circular imports if needed
        from system_config import GMAIL_ACCOUNTS, GMAIL_DEFAULT

        if not GMAIL_ACCOUNTS:
            return "Error: No Gmail accounts configured. Please set JARVIS_GMAIL_ACCOUNTS in .env"

        # GMAIL_ACCOUNTS is {email_or_nickname: app_password}
        account_key = (account or "").strip().lower()
        if account_key:
            if account_key not in GMAIL_ACCOUNTS:
                # Allow matching by email substring
                matches = [k for k in GMAIL_ACCOUNTS if account_key in k.lower()]
                if not matches:
                    available = ", ".join(GMAIL_ACCOUNTS.keys())
                    return f"Error: Account '{account}' not found. Available accounts: {available}"
                account_key = matches[0]
            email = account_key
            app_password = GMAIL_ACCOUNTS[account_key]
        else:
            default_key = (GMAIL_DEFAULT or "").strip().lower()
            if default_key and default_key in GMAIL_ACCOUNTS:
                email = default_key
                app_password = GMAIL_ACCOUNTS[default_key]
            else:
                email, app_password = next(iter(GMAIL_ACCOUNTS.items()))
                # If key looks like an email use it; value is always the app password
                if "@" not in email and ":" in str(app_password):
                    # Legacy mistaken "email:password" packed in value
                    email, app_password = str(app_password).split(":", 1)

        # Create message
        msg = MIMEMultipart()
        msg['From'] = email
        msg['To'] = to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Connect to Gmail SMTP
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email, app_password)
        text = msg.as_string()
        server.sendmail(email, to, text)
        server.quit()

        return f"Email sent from {email} to {to}"

    except smtplib.SMTPAuthenticationError:
        return "Error: Gmail authentication failed. Check your email and app password."
    except smtplib.SMTPException as e:
        return f"Error sending email: SMTP error - {str(e)}"
    except Exception as e:
        return f"Error sending email: {str(e)}"

@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Perform a web search using DuckDuckGo.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 3).

    Returns:
        A short numbered list of search results in the format "title - snippet".
        Returns an error message if search fails.
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)

        if not results:
            return f"No results found for query: {query}"

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            snippet = result.get('body', 'No snippet available')
            formatted_results.append(f"{i}. {title} - {snippet}")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error performing web search for '{query}': {str(e)}"