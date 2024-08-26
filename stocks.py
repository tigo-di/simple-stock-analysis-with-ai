#Import da libs
import json
import os
from datetime import datetime

import yfinance as yf


from crewai.agent import Agent
from crewai.task import Task
from crewai.crew import Crew
from crewai.process import Process

from langchain.tools import Tool

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# CRIANDO YAhOO FINANCE TOOL
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-20", end="2024-08-20")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "This tool fetches stock prices for {ticket} from the last year about a specific company from Yahoo Finance API.",
    func= lambda ticket: fetch_stock_price(ticket)
)

# IMPORTANDO OPENAI LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")

# CRIANDO AGENTE

stockPriceAnalyst = Agent(
  role="Senior stock Price Analyst",
  goal="Find the {ticket} stock price for a specific company and analyses trends.",
  backstory="""You are a highly experienced in analyzing the price of a specific stock and make predictions about its future price. Your job is to provide insights on stock prices and trends to help clients make informed investment""",
  verbose=True,
  llm=llm,
  max_iter=5,
  memory=True,
  tools=[yahoo_finance_tool],
)

# CRIANDO TAREFA

getPriceTask = Task(
  name="Stock Price Analysis",
  description="Analyze the stock price history of {ticket} for a specific company and provide insights on trends analyses of up, down or sideways.",
  expected_output="""Specify the current trend stock price - up, down or sideways.
  eg. stock= 'APPL, price UP'
  """,
  agent= stockPriceAnalyst
)

# IMPORTANDO FERRAMENTA DE PESQUISA

search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

# CRIANDO AGENTE DE NOTÍCIAS
stockNewsAnalyst = Agent(
  role="Stock News Analyst",
  goal="""Creat a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down or sideways with the news context. For each request stock asset, specify a numbet between 0 and 100, where 0 is extreme fear and 100 is extreme greed""",
  backstory="""You are a highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years. You're also master level analyts in the tradicional markets and have deep understanding of human psychology and market sentiment.
  you undestand news, theirs titles and information, but you look at those with a health dose of skepticism.
  You consider also the source of the news articles.""",
  verbose=True,
  llm=llm,
  max_iter=10,
  memory=True,
  allow_deletion=False,
  tools=[search_tool],
)

# CRIANDO TAREFA DE PEGAR AS NOTÍCIAS

getNewsTask = Task(
  name="Task Get News",
  description=f"""Take the stock and always include BTC to it (if not request).
  Use the search tool to search each one individually.
  The current date is {datetime.now()}.
  Compose the results into a helpfull report""",
  expected_output="""A summary of the overeal market and one sentence summary for each request asset.
  Include a fear/greed score for each asset based on the news sentiment. Use format:
  <STOCK ASSET>
  <SUMMARY BASED ON NEWS>
  <TREND PREDICTION>
  <FEAR/GREED SCORE>
  """,
  agent=stockNewsAnalyst,
)

stockWriteAnalyst = Agent(
  role = "Senior Stock Analyts Writer",
  goal = """Analyze the trends price and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.""",
  backstory = """You're widely accepted as the best stock analyst in the market. You understand complex concepts and creat compelling stories and naratives that resonate with wider audiences.

  You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. You're able to hold multiple opinions when analyzing anything.""",
  verbose=True,
  llm=llm,
  max_iter=5,
  memory=True,
  allow_deletion=True,
)

getWriteTask = Task(
  description = """Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company that is brief and highlights the most important points.
  Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
  Include the previous analyses of stock trend and news summary.
  """,
  expected_output = """An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner.
  It should contain:
  - 3 bullets executive summary
  - Introduction - set the overall picture and spike up the interest
  - main part provides the meat of the analysis including the news summary and fear/greed scores
  - summary - key facts and concret future trend prediction - up, down or sideways.""",
  agent = stockWriteAnalyst,
  context = [getPriceTask, getNewsTask],
)

crew = Crew(
  name = "Stock Market Crew",
  agents = [stockPriceAnalyst, stockNewsAnalyst, stockWriteAnalyst],
  tasks = [getPriceTask, getNewsTask, getWriteTask],
  verbose = True,
  process = Process.hierarchical,
  full_output = True,
  share_crew = False,
  manager_llm = llm,
  max_iter = 15,
)

# results = crew.kickoff(inputs={'ticket': 'AAPL'})

# results['final_output']

with st.sidebar:
  st.header('Enter the ticket stock')

  with st.form(key='research_form'):
    topic = st.text_input("Select the ticket")
    submit_button = st.form_submit_button(label = 'Run Research')

if submit_button:
  if not topic:
    st.error('Please fill the ticket field')
  else:
    results = crew.kickoff(inputs={'ticket': topic})
    st.subheader('Results of your research:')
    st.write(results['final_output'])
