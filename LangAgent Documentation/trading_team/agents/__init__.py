# LangAgent/trading_team/agents/__init__.py

from .fundamentals_analyst import make_fundamentals_agent as fundamentals_analyst
from .sentiments_analyst import make_sentiments_agent as sentiments_analyst
from .valuations_analyst import make_valuations_agent as valuations_analyst
from .technicals_analyst import make_technicals_agent as technicals_analyst
from .risk_manager import make_risk_manager_agent as risk_manager
from .portfolio_manager import make_portfolio_manager_agent as portfolio_manager

# Define what will be accessible when doing `from LangAgent.trading_team.agents import *`
__all__ = ['fundamentals_analyst', 'sentiments_analyst', 'valuations_analyst', 'technicals_analyst', 'risk_manager', 'portfolio_manager']
