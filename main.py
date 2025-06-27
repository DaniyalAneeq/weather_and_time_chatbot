from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from dotenv import load_dotenv
import os
import requests
from datetime import datetime
import chainlit as cl
from typing import cast
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()

set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

@cl.on_chat_start
async def start():

    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client,
    )
    
    cl.user_session.set("user_chat_history", [])
    
    @function_tool
    def get_current_weather(city: str) -> str:
        """
        Get the current weather in the specified city.
        """
    
        weather_api_key = os.getenv("WEATHER_API_KEY")
    
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
        
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data["cod"] != 200:
                return f"Error from API: {data.get('message', 'Unknown error')}"

            weather = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            return f"The current weather in {city.title()} is {weather} with a temperature of {temperature}°C."
        except Exception as e:
            return f"Error occurred: {str(e)}"

    @function_tool
    def get_current_time(city: str) -> str:
        """
        Get the current time in the specified city.
        """
        timezone_api_key=os.getenv("TimeZone_API_KEY")
    
        city_to_timezone ={
            "karachi": "Asia/Karachi",
            "new york": "America/New_York",
            "london": "Europe/London",
            "tokyo": "Asia/Tokyo",
            "sydney": "Australia/Sydney",
            "paris": "Europe/Paris",
            "mumbai": "Asia/Kolkata",
            "beijing": "Asia/Shanghai",
        }
    
        city_key = city.lower()
    
        if city_key not in city_to_timezone:
            return f"Sorry, I don't have the current time for {city}."
    
        timezone = city_to_timezone[city_key]
        url= f"http://api.timezonedb.com/v2.1/get-time-zone?key{timezone_api_key}&format=json&by=zone&zone={timezone}"
    
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
        
            if data["status"] != "OK":
                return f"Error from API: {data.get('message', 'Unknown error')}"
        
            # Parse the datetime string and format to 12-hour clock with AM/PM
            time_str = data["formatted"]
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            formatted_time = dt.strftime("%I:%M %p")
        
            return f"The current time in {city.title()} is {formatted_time}"
        
        except Exception as e:
            return f"Error occurred: {str(e)}"
    
    
    agent: Agent = Agent(
        name="Assistant",
        instructions="""
        Your task is to assist the user with their questions and provide helpful information. if user ask for the current time in the provided city, then you should call to get_current_time function with the city name. If user asks for the current weather in the provided city, then you should call to get_current_weather function with the city name. Note that If user ask for the irrelevant question then you should respond with 'I am not sure about that. Sorry! I'm designed to fetch real-time weather updates and time around the world!'. 
        """,
        model=model,
        tools=[get_current_time, get_current_weather]
    )
    
    cl.user_session.set("agent", agent)
    
    await cl.Message(content="Welcome to my Weather and Timezone Agent!").send()

@cl.on_message
async def main(message:cl.Message):
    
    msg = cl.Message(content="Thinking...")
    await msg.send()
    
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    
    user_history = cl.user_session.get("user_chat_history") or []
    
    user_history.append({"role":"user", "content":message.content})
    cl.user_session.set("user_chat_history", user_history)
    
    steam_message = cl.Message(content="")
    await steam_message.send()
    
    try:
        
        result = Runner.run_streamed(
            starting_agent= agent,
            input=user_history,
        )
        
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                
                # Clear the "Thinking..." message once streaming starts
                if steam_message.content == "":
                    await msg.remove()
                    
                await steam_message.stream_token(event.data.delta)
        
        response_agent = result.final_output
        
        steam_message.content = response_agent
        await steam_message.update()
        
        cl.user_session.set("agent_chat_history", result.to_input_list())
            
        print(f"User: {message.content}")
        print(f"Assistant: {response_agent}")
        
    except Exception as e:
        # Remove the "Thinking..." message in case of an error
        await msg.remove()
        steam_message.content = f"Error: {str(e)}"
        await steam_message.update()
        print(f"Error: {str(e)}")
