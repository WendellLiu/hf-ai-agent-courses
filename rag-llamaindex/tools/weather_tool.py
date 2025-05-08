import os
from llama_index.core.tools import FunctionTool


# Code Agent Example: Retrieve Weather Information
def get_weather(city):
    """Fetches the current weather information for a given city."""
    import requests

    weather_api_key = os.getenv("WEATHER_API_KEY")

    api_url = f"https://api.weatherapi.com/v1/forecast.json?key={weather_api_key}&q={city}&days=3&aqi=no&alerts=no"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return "Error: Unable to fetch weather data."


weather_tool = FunctionTool.from_defaults(get_weather)
