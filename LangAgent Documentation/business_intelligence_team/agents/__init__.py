# LangAgent/business_intelligence_team/agents/__init__.py

from .bi_analyst_agent import create_bi_analyst as bi_analyst

# Define what will be accessible when doing `from LangAgent.business_intelligence_team.agents import *`
__all__ = ['bi_analyst']
