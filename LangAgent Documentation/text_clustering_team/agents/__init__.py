# LangAgent/text_clustering_team/agents/__init__.py

from .topic_generator import create_topic_generator as topic_generator

# Define what will be accessible when doing `from LangAgent.text_clustering_team.agents import *`
__all__ = ['topic_generator']
