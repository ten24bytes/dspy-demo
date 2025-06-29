#!/usr/bin/env python3
"""
AI Text Game with DSPy

This script demonstrates how to build an interactive text-based game using DSPy.
It features dynamic story generation, character interactions, and game state management.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from utils import setup_default_lm, print_step, print_result, print_error

@dataclass
class GameState:
    """Represents the current state of the game."""
    location: str = "forest clearing"
    inventory: List[str] = field(default_factory=list)
    health: int = 100
    experience: int = 0
    level: int = 1
    story_context: List[str] = field(default_factory=list)
    visited_locations: List[str] = field(default_factory=list)
    game_flags: Dict[str, bool] = field(default_factory=dict)
    turn_count: int = 0

@dataclass
class Character:
    """Represents a game character."""
    name: str
    description: str
    dialogue_style: str
    location: str
    friendly: bool = True
    conversation_history: List[str] = field(default_factory=list)

@dataclass
class GameAction:
    """Represents an action taken in the game."""
    action_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    result_message: str = ""

class StoryGeneration(dspy.Signature):
    """Generate engaging story content for the game."""
    
    current_location = dspy.InputField(desc="Current game location")
    player_action = dspy.InputField(desc="Action taken by the player")
    game_context = dspy.InputField(desc="Current game state and context")
    story_history = dspy.InputField(desc="Recent story events")
    story_continuation = dspy.OutputField(desc="Engaging story continuation based on player action")
    new_opportunities = dspy.OutputField(desc="New opportunities or choices available to the player")

class ActionValidation(dspy.Signature):
    """Validate and interpret player actions."""
    
    player_input = dspy.InputField(desc="Raw player input/command")
    current_location = dspy.InputField(desc="Current location description")
    available_items = dspy.InputField(desc="Items available in current location")
    player_inventory = dspy.InputField(desc="Items in player inventory")
    action_type = dspy.OutputField(desc="Type of action: move, take, use, talk, examine, attack, etc.")
    action_target = dspy.OutputField(desc="Target of the action if applicable")
    is_valid = dspy.OutputField(desc="Whether the action is valid in current context")
    explanation = dspy.OutputField(desc="Explanation of action validity and effects")

class CharacterDialogue(dspy.Signature):
    """Generate character dialogue and responses."""
    
    character_name = dspy.InputField(desc="Name of the character")
    character_personality = dspy.InputField(desc="Character's personality and style")
    player_message = dspy.InputField(desc="What the player said to the character")
    conversation_history = dspy.InputField(desc="Previous conversation context")
    game_context = dspy.InputField(desc="Current game state and location")
    character_response = dspy.OutputField(desc="Character's response to the player")
    character_emotion = dspy.OutputField(desc="Character's emotional state")
    new_information = dspy.OutputField(desc="Any new information revealed by character")

class LocationDescription(dspy.Signature):
    """Generate detailed location descriptions."""
    
    location_name = dspy.InputField(desc="Name of the location")
    time_of_day = dspy.InputField(desc="Time context (morning, evening, etc.)")
    weather = dspy.InputField(desc="Weather conditions")
    player_actions = dspy.InputField(desc="Recent player actions affecting the location")
    visited_before = dspy.InputField(desc="Whether player has been here before")
    description = dspy.OutputField(desc="Rich, immersive description of the location")
    notable_features = dspy.OutputField(desc="Specific features the player can interact with")
    atmosphere = dspy.OutputField(desc="Mood and atmosphere of the location")

class GameMaster(dspy.Module):
    """Main game master that orchestrates the game experience."""
    
    def __init__(self):
        super().__init__()
        self.story_generator = dspy.ChainOfThought(StoryGeneration)
        self.action_validator = dspy.ChainOfThought(ActionValidation)
        self.location_describer = dspy.ChainOfThought(LocationDescription)
        
        # Game world
        self.locations = self._initialize_locations()
        self.characters = self._initialize_characters()
        self.items = self._initialize_items()
        
        # Game state
        self.state = GameState()
        self.state.visited_locations.append(self.state.location)
    
    def _initialize_locations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize game world locations."""
        return {
            "forest clearing": {
                "description": "A peaceful clearing surrounded by tall oak trees",
                "items": ["wooden stick", "shiny stone"],
                "exits": ["north:dark forest", "east:old village", "south:misty lake"],
                "characters": ["forest sprite"]
            },
            "dark forest": {
                "description": "A dense forest where sunlight barely penetrates",
                "items": ["mysterious mushroom", "ancient scroll"],
                "exits": ["south:forest clearing", "west:mountain path"],
                "characters": ["wise owl"]
            },
            "old village": {
                "description": "A quaint village with cobblestone streets and thatched roofs",
                "items": ["healing potion", "copper coin"],
                "exits": ["west:forest clearing", "north:village inn"],
                "characters": ["village elder", "merchant"]
            },
            "misty lake": {
                "description": "A serene lake shrouded in morning mist",
                "items": ["crystal water", "silver fish"],
                "exits": ["north:forest clearing", "east:hidden cave"],
                "characters": ["lake guardian"]
            },
            "village inn": {
                "description": "A warm, cozy inn with a crackling fireplace",
                "items": ["traveler's map", "hot meal"],
                "exits": ["south:old village"],
                "characters": ["innkeeper", "mysterious traveler"]
            },
            "hidden cave": {
                "description": "A mysterious cave hidden behind the waterfall",
                "items": ["ancient treasure", "glowing crystal"],
                "exits": ["west:misty lake"],
                "characters": ["cave hermit"]
            }
        }
    
    def _initialize_characters(self) -> Dict[str, Character]:
        """Initialize game characters."""
        return {
            "forest sprite": Character(
                "Forest Sprite",
                "A magical being that protects the forest",
                "Playful and mysterious, speaks in riddles",
                "forest clearing"
            ),
            "wise owl": Character(
                "Wise Owl",
                "An ancient owl with knowledge of the forest's secrets",
                "Wise and cryptic, gives helpful advice",
                "dark forest"
            ),
            "village elder": Character(
                "Village Elder",
                "The respected leader of the village",
                "Formal and knowledgeable about local history",
                "old village"
            ),
            "merchant": Character(
                "Merchant",
                "A traveling trader with exotic goods",
                "Enthusiastic and business-minded",
                "old village"
            ),
            "lake guardian": Character(
                "Lake Guardian",
                "A mystical being that watches over the lake",
                "Serene and protective, speaks of ancient times",
                "misty lake"
            ),
            "innkeeper": Character(
                "Innkeeper",
                "A friendly host who knows all the local gossip",
                "Warm and talkative, loves sharing stories",
                "village inn"
            ),
            "mysterious traveler": Character(
                "Mysterious Traveler",
                "A cloaked figure with secrets to hide",
                "Enigmatic and cautious, speaks carefully",
                "village inn"
            ),
            "cave hermit": Character(
                "Cave Hermit",
                "A reclusive sage living in the hidden cave",
                "Wise but eccentric, offers cryptic guidance",
                "hidden cave"
            )
        }
    
    def _initialize_items(self) -> Dict[str, Dict[str, Any]]:
        """Initialize game items."""
        return {
            "wooden stick": {"description": "A sturdy oak branch", "type": "weapon", "value": 5},
            "shiny stone": {"description": "A smooth stone that gleams in the light", "type": "misc", "value": 2},
            "mysterious mushroom": {"description": "A glowing mushroom with unknown properties", "type": "consumable", "value": 10},
            "ancient scroll": {"description": "A weathered scroll with cryptic writing", "type": "quest", "value": 15},
            "healing potion": {"description": "A red potion that restores health", "type": "consumable", "value": 20},
            "copper coin": {"description": "A tarnished copper coin", "type": "currency", "value": 1},
            "crystal water": {"description": "Pure water from the mystical lake", "type": "consumable", "value": 8},
            "silver fish": {"description": "A magical fish that grants wishes", "type": "quest", "value": 25},
            "traveler's map": {"description": "A detailed map of the surrounding areas", "type": "tool", "value": 12},
            "hot meal": {"description": "A warm, nourishing meal", "type": "consumable", "value": 6},
            "ancient treasure": {"description": "A chest filled with gold and gems", "type": "treasure", "value": 100},
            "glowing crystal": {"description": "A crystal pulsing with magical energy", "type": "magic", "value": 50}
        }
    
    def process_player_input(self, player_input: str) -> Dict[str, Any]:
        """Process player input and update game state."""
        
        # Validate and interpret action
        current_location_data = self.locations.get(self.state.location, {})
        
        validation_result = self.action_validator(
            player_input=player_input,
            current_location=current_location_data.get("description", ""),
            available_items=str(current_location_data.get("items", [])),
            player_inventory=str(self.state.inventory)
        )
        
        if validation_result.is_valid.lower() == "true":
            action = GameAction(
                action_type=validation_result.action_type,
                description=player_input,
                success=True,
                result_message=validation_result.explanation
            )
            
            # Execute the action
            self._execute_action(action, validation_result.action_target)
            
            # Generate story continuation
            story_result = self.story_generator(
                current_location=self.state.location,
                player_action=player_input,
                game_context=self._get_game_context(),
                story_history=str(self.state.story_context[-3:])
            )
            
            self.state.story_context.append(story_result.story_continuation)
            self.state.turn_count += 1
            
            return {
                "success": True,
                "action": action,
                "story": story_result.story_continuation,
                "opportunities": story_result.new_opportunities,
                "game_state": self._get_game_status()
            }
        
        else:
            return {
                "success": False,
                "message": validation_result.explanation,
                "suggestion": "Try a different action or be more specific."
            }
    
    def _execute_action(self, action: GameAction, target: str):
        """Execute a validated game action."""
        
        action_type = action.action_type.lower()
        current_location_data = self.locations.get(self.state.location, {})
        
        if action_type == "move":
            self._handle_movement(target)
        elif action_type == "take":
            self._handle_take_item(target)
        elif action_type == "use":
            self._handle_use_item(target)
        elif action_type == "examine":
            self._handle_examine(target)
        elif action_type == "talk":
            self._handle_talk_to_character(target)
        # Add more action types as needed
    
    def _handle_movement(self, direction: str):
        """Handle player movement between locations."""
        current_location_data = self.locations.get(self.state.location, {})
        exits = current_location_data.get("exits", [])
        
        for exit_info in exits:
            if ":" in exit_info:
                exit_direction, destination = exit_info.split(":", 1)
                if direction.lower() in exit_direction.lower():
                    self.state.location = destination
                    if destination not in self.state.visited_locations:
                        self.state.visited_locations.append(destination)
                    break
    
    def _handle_take_item(self, item: str):
        """Handle taking items from location."""
        current_location_data = self.locations.get(self.state.location, {})
        available_items = current_location_data.get("items", [])
        
        for available_item in available_items:
            if item.lower() in available_item.lower():
                self.state.inventory.append(available_item)
                available_items.remove(available_item)
                self.state.experience += 5
                break
    
    def _handle_use_item(self, item: str):
        """Handle using items from inventory."""
        for inv_item in self.state.inventory:
            if item.lower() in inv_item.lower():
                item_data = self.items.get(inv_item, {})
                item_type = item_data.get("type", "misc")
                
                if item_type == "consumable":
                    if "healing" in inv_item.lower():
                        self.state.health = min(100, self.state.health + 25)
                    self.state.inventory.remove(inv_item)
                    self.state.experience += 10
                break
    
    def _handle_examine(self, target: str):
        """Handle examining objects or locations."""
        # This would provide detailed descriptions
        self.state.experience += 2
    
    def _handle_talk_to_character(self, character_name: str):
        """Handle conversations with characters."""
        # This would trigger dialogue generation
        self.state.experience += 8
    
    def get_location_description(self) -> str:
        """Get detailed description of current location."""
        
        visited_before = self.state.location in self.state.visited_locations[:-1]
        
        description_result = self.location_describer(
            location_name=self.state.location,
            time_of_day="midday",  # Could be dynamic
            weather="clear",  # Could be dynamic
            player_actions=str(self.state.story_context[-2:]),
            visited_before=str(visited_before)
        )
        
        return f"{description_result.description}\n\n{description_result.notable_features}\n\nAtmosphere: {description_result.atmosphere}"
    
    def _get_game_context(self) -> str:
        """Get current game context as string."""
        return f"Location: {self.state.location}, Health: {self.state.health}, Level: {self.state.level}, Inventory: {self.state.inventory}"
    
    def _get_game_status(self) -> Dict[str, Any]:
        """Get current game status."""
        return {
            "location": self.state.location,
            "health": self.state.health,
            "level": self.state.level,
            "experience": self.state.experience,
            "inventory": self.state.inventory,
            "turn_count": self.state.turn_count
        }

class CharacterInteraction(dspy.Module):
    """Handle character interactions and dialogue."""
    
    def __init__(self):
        super().__init__()
        self.dialogue_generator = dspy.ChainOfThought(CharacterDialogue)
    
    def talk_to_character(self, character: Character, player_message: str, game_context: str) -> Dict[str, Any]:
        """Generate character dialogue response."""
        
        dialogue_result = self.dialogue_generator(
            character_name=character.name,
            character_personality=character.dialogue_style,
            player_message=player_message,
            conversation_history=str(character.conversation_history[-3:]),
            game_context=game_context
        )
        
        # Update conversation history
        character.conversation_history.append(f"Player: {player_message}")
        character.conversation_history.append(f"{character.name}: {dialogue_result.character_response}")
        
        return {
            "response": dialogue_result.character_response,
            "emotion": dialogue_result.character_emotion,
            "new_information": dialogue_result.new_information,
            "character": character.name
        }

def demonstrate_text_game():
    """Demonstrate the AI text game."""
    
    print_step("AI Text Adventure Game")
    
    # Initialize game
    game = GameMaster()
    character_system = CharacterInteraction()
    
    print_result("=" * 50)
    print_result("Welcome to the AI Text Adventure!")
    print_result("=" * 50)
    
    # Show initial location
    print_result("\nYou find yourself in a mystical realm...")
    location_desc = game.get_location_description()
    print_result(f"\n{location_desc}")
    
    # Game loop simulation (normally would be interactive)
    sample_commands = [
        "look around",
        "take wooden stick",
        "examine shiny stone",
        "go north",
        "talk to wise owl",
        "take ancient scroll",
        "go south",
        "go east",
        "use healing potion"
    ]
    
    print_result(f"\nInitial Status: {game._get_game_status()}")
    print_result("\n" + "=" * 50)
    print_result("Simulating Player Actions:")
    print_result("=" * 50)
    
    for i, command in enumerate(sample_commands[:6], 1):  # Limit for demo
        print_result(f"\nTurn {i}: Player enters '{command}'")
        
        result = game.process_player_input(command)
        
        if result["success"]:
            print_result(f"Story: {result['story']}")
            print_result(f"New Opportunities: {result['opportunities']}")
            print_result(f"Status: {result['game_state']}")
            
            # Show location description if moved
            if "move" in result["action"].action_type.lower():
                print_result(f"\nNew Location:")
                print_result(game.get_location_description())
        else:
            print_result(f"Invalid Action: {result['message']}")
            print_result(f"Suggestion: {result['suggestion']}")
        
        print_result("-" * 30)

def demonstrate_character_interaction():
    """Demonstrate character dialogue system."""
    
    print_step("Character Interaction System")
    
    character_system = CharacterInteraction()
    
    # Create a character
    forest_sprite = Character(
        "Forest Sprite",
        "A magical being that protects the forest",
        "Playful and mysterious, speaks in riddles",
        "forest clearing"
    )
    
    # Simulate conversation
    player_messages = [
        "Hello, who are you?",
        "Can you help me find treasure?",
        "What do you know about this forest?"
    ]
    
    game_context = "Player is in forest clearing, has wooden stick, exploring the world"
    
    for message in player_messages:
        print_result(f"\nPlayer: {message}")
        
        response = character_system.talk_to_character(
            forest_sprite, 
            message, 
            game_context
        )
        
        print_result(f"{response['character']}: {response['response']}")
        print_result(f"Emotion: {response['emotion']}")
        if response['new_information']:
            print_result(f"New Info: {response['new_information']}")

def create_custom_game_scenario():
    """Create a custom game scenario."""
    
    print_step("Custom Game Scenario")
    
    game = GameMaster()
    
    # Set up a specific scenario
    game.state.location = "hidden cave"
    game.state.inventory = ["ancient scroll", "glowing crystal"]
    game.state.experience = 50
    game.state.level = 2
    
    print_result("Custom Scenario: The Cave of Mysteries")
    print_result(f"Current Status: {game._get_game_status()}")
    
    # Get location description
    location_desc = game.get_location_description()
    print_result(f"\nLocation: {location_desc}")
    
    # Test specific actions in this scenario
    scenario_actions = [
        "examine glowing crystal",
        "use ancient scroll",
        "talk to cave hermit"
    ]
    
    for action in scenario_actions:
        print_result(f"\nAction: {action}")
        result = game.process_player_input(action)
        
        if result["success"]:
            print_result(f"Story: {result['story']}")
            print_result(f"Opportunities: {result['opportunities']}")
        else:
            print_result(f"Issue: {result['message']}")

def main():
    """Main function demonstrating the AI text game."""
    
    print("=" * 60)
    print("DSPy AI Text Game Demo")
    print("=" * 60)
    
    # Setup language model
    lm = setup_default_lm()
    if not lm:
        return
    
    try:
        # Main game demonstration
        demonstrate_text_game()
        
        # Character interaction
        demonstrate_character_interaction()
        
        # Custom scenario
        create_custom_game_scenario()
        
        print_step("AI Text Game Demo Complete!")
        print_result("\nIn a real implementation, this would be fully interactive with:")
        print_result("- Continuous player input")
        print_result("- Save/load game states")
        print_result("- Dynamic world events")
        print_result("- Character relationship tracking")
        print_result("- Quest system")
        print_result("- Combat mechanics")
        
    except Exception as e:
        print_error(f"Error in text game demo: {e}")

if __name__ == "__main__":
    main()
