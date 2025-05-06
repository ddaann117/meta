 # Check for anomalies
            message_vector = self.encoder.encode(ual_message)
            anomaly_status = self.anomaly_detector.detect_anomaly(message_vector)
            
            if anomaly_status == "Red":
                self.logger.warning(f"Red anomaly detected in message from {source_id} to {target_id}")
                return {
                    "status": "rejected",
                    "reason": "Anomaly detected",
                    "anomaly_level": "Red"
                }
            
            # Translate for target system
            translated = self.universal_language.translate_between_systems(
                ual_message, source_id, target_id
            )
            
            # Record translation in history
            self.translation_history.append({
                "timestamp": time.time(),
                "source_id": source_id,
                "target_id": target_id,
                "message_id": ual_message.get("header", {}).get("message_id", "unknown"),
                "intent": intent,
                "translated": bool(translated and "error" not in translated)
            })
            
            # Update message counters
            source_system["messages"]["sent"] += 1
            target_system["messages"]["received"] += 1
            target_system["last_activity"] = time.time()
            
            if translated and "error" not in translated:
                source_system["messages"]["translated"] += 1
            
            # Delivery status
            return {
                "status": "delivered",
                "message_id": ual_message.get("header", {}).get("message_id", "unknown"),
                "anomaly_status": anomaly_status,
                "translated": bool(translated and "error" not in translated),
                "target_format": target_system["type"]
            }
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def translate_content(self, content: Any, source_type: str, 
                        target_type: str) -> Dict[str, Any]:
        """
        Translate content between different AI representations
        
        Args:
            content: Content to translate
            source_type: Source representation type
            target_type: Target representation type
            
        Returns:
            Translated content
        """
        try:
            # Use protocol translator for direct translation
            direct_translation = self.translator.translate(
                content, source_type, target_type
            )
            
            # Also use the universal language for comparison
            # First, register temporary systems if needed
            source_id = f"temp_{source_type}_{int(time.time())}"
            target_id = f"temp_{target_type}_{int(time.time())}"
            
            self.universal_language.register_ai_system(
                source_id, source_type, ["translation"]
            )
            
            self.universal_language.register_ai_system(
                target_id, target_type, ["translation"]
            )
            
            # Create UAL message
            ual_message = self.universal_language.create_message(
                content=content,
                intent="INFORM",
                sender=source_id,
                receiver=target_id
            )
            
            # Translate through UAL
            ual_translation = self.universal_language.translate_between_systems(
                ual_message, source_id, target_id
            )
            
            return {
                "direct_translation": direct_translation,
                "ual_translation": ual_translation,
                "source_type": source_type,
                "target_type": target_type
            }
            
        except Exception as e:
            self.logger.error(f"Error translating content: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def analyze_communication(self, system_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze communication patterns for a specific system or all systems
        
        Args:
            system_id: Optional system ID to analyze (all systems if None)
            
        Returns:
            Analysis results
        """
        try:
            results = {
                "timestamp": time.time(),
                "systems": {}
            }
            
            # Filter systems to analyze
            systems_to_analyze = {}
            if system_id:
                if system_id in self.connected_systems:
                    systems_to_analyze[system_id] = self.connected_systems[system_id]
                else:
                    return {"status": "error", "error": f"System not found: {system_id}"}
            else:
                systems_to_analyze = self.connected_systems
            
            # Calculate basic statistics
            total_messages = 0
            total_translations = 0
            system_types = set()
            
            # Analyze each system
            for sys_id, system in systems_to_analyze.items():
                system_types.add(system["type"])
                
                msgs = system["messages"]
                total_sent = msgs.get("sent", 0)
                total_received = msgs.get("received", 0)
                total_translated = msgs.get("translated", 0)
                
                total_messages += total_sent + total_received
                total_translations += total_translated
                
                # Calculate activity level
                now = time.time()
                last_activity = system.get("last_activity", 0)
                hours_since_activity = (now - last_activity) / 3600
                
                if hours_since_activity < 1:
                    activity_level = "high"
                elif hours_since_activity < 24:
                    activity_level = "medium"
                else:
                    activity_level = "low"
                
                # System-specific analysis
                results["systems"][sys_id] = {
                    "type": system["type"],
                    "messages_sent": total_sent,
                    "messages_received": total_received,
                    "translations": total_translated,
                    "activity_level": activity_level,
                    "hours_since_activity": round(hours_since_activity, 2),
                    "capabilities": system.get("capabilities", [])
                }
            
            # Overall statistics
            results["total_messages"] = total_messages
            results["total_translations"] = total_translations
            results["system_types"] = list(system_types)
            results["system_count"] = len(systems_to_analyze)
            
            # Add HMM state information
            results["hmm_state"] = {
                "current_state": self.hmm.get_current_state_index(),
                "entropy": self.hmm.current_entropy
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing communication: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def demonstrate(self) -> Dict[str, Any]:
        """
        Run a complete demonstration of the system capabilities
        
        Returns:
            Demonstration results
        """
        try:
            results = {
                "timestamp": time.time(),
                "components": {},
                "translations": {},
                "analysis": {}
            }
            
            # Step 1: Register sample systems
            system_types = ["llm", "diffusion", "rl", "recommender"]
            for system_type in system_types:
                system_id = f"demo_{system_type}"
                reg_result = self.register_ai_system(
                    system_id, 
                    system_type, 
                    ["text", "universal", system_type]
                )
                results["components"][system_id] = {
                    "registration": reg_result.get("status"),
                    "type": system_type
                }
            
            # Step 2: Test messages between different systems
            test_cases = [
                ("demo_llm", "demo_diffusion", "A beautiful sunset over mountains"),
                ("demo_diffusion", "demo_llm", {"image_type": "landscape", "description": "Ocean waves crashing on rocky shore"}),
                ("demo_llm", "demo_rl", "Move three steps forward then turn right"),
                ("demo_rl", "demo_llm", {"action": "pick_up", "object": "book", "confidence": 0.9}),
                ("demo_recommender", "demo_llm", {"items": ["movie_123", "book_456"], "scores": [0.95, 0.87]}),
                ("demo_llm", "demo_recommender", "I prefer science fiction and mystery genres")
            ]
            
            for source_id, target_id, content in test_cases:
                result = self.send_message(source_id, target_id, content)
                case_key = f"{source_id}_to_{target_id}"
                results["translations"][case_key] = {
                    "content": str(content)[:50] + ("..." if len(str(content)) > 50 else ""),
                    "status": result.get("status"),
                    "translated": result.get("translated", False)
                }
            
            # Step 3: Generate analysis
            results["analysis"] = self.analyze_communication()
            
            # Step 4: Demonstrate UAL capabilities
            ual_demo = self.universal_language.demonstrate_translation()
            results["ual_demonstration"] = {
                "status": ual_demo.get("status"),
                "translation_count": len(ual_demo.get("demo_results", {})),
                "supported_concepts": len(self.universal_language.list_concepts())
            }
            
            # Step 5: Show state information
            hmm_state = self.hmm.get_current_state_index()
            hmm_entropy = self.hmm.current_entropy
            
            results["state_info"] = {
                "hmm_state": hmm_state,
                "hmm_entropy": hmm_entropy,
                "anomaly_status": "Green",  # Assuming normal operation for demo
                "system_uptime": time.time() - results["timestamp"]
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in demonstration: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def save_state(self, filepath: Optional[str] = None) -> bool:
        """
        Save the current system state
        
        Args:
            filepath: Optional filepath (uses default if None)
            
        Returns:
            Success flag
        """
        try:
            if filepath is None:
                filepath = self.config.STATE_FILE
            
            # Use the RecursiveWatchDog to save core components state
            watchdog_saved = self.watchdog.save_state(filepath)
            
            # Create a state dictionary for additional components
            state_dict = {
                "connected_systems": self.connected_systems,
                "translation_history": list(self.translation_history),
                "timestamp": time.time(),
                "version": self.universal_language.get_version()
            }
            
            # Save additional state
            additional_state_file = f"{os.path.splitext(filepath)[0]}_additional.pkl"
            with open(additional_state_file, 'wb') as f:
                pickle.dump(state_dict, f)
            
            # Save self-reference
            self.self_ref_manager.save_description()
            
            self.logger.info(f"System state saved to {filepath} and {additional_state_file}")
            return watchdog_saved
            
        except Exception as e:
            self.logger.error(f"Error saving system state: {e}", exc_info=True)
            return False
    
    def load_state(self, filepath: Optional[str] = None) -> bool:
        """
        Load system state from file
        
        Args:
            filepath: Optional filepath (uses default if None)
            
        Returns:
            Success flag
        """
        try:
            if filepath is None:
                filepath = self.config.STATE_FILE
            
            # Use the RecursiveWatchDog to load core components state
            watchdog_loaded = self.watchdog.load_state(filepath)
            
            # Try to load additional state
            additional_state_file = f"{os.path.splitext(filepath)[0]}_additional.pkl"
            if os.path.exists(additional_state_file):
                with open(additional_state_file, 'rb') as f:
                    state_dict = pickle.load(f)
                
                self.connected_systems = state_dict.get("connected_systems", {})
                self.translation_history = deque(
                    state_dict.get("translation_history", []),
                    maxlen=100
                )
                
                self.logger.info(f"Additional system state loaded from {additional_state_file}")
            
            self.logger.info(f"System state loaded from {filepath}")
            return watchdog_loaded
            
        except Exception as e:
            self.logger.error(f"Error loading system state: {e}", exc_info=True)
            return False

# =================================================================
# Recursive Watch Dog (Main Orchestrator)
# =================================================================

class RecursiveWatchDog:
    """
    Main orchestrator class for the SDR-DBHMM-WD system.
    
    Coordinates all components, manages the flow of data, and makes high-level
    decisions about when to trigger refinement processes or anomaly responses.
    """
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.self_ref_manager = SelfReferenceManager(config)
        self.encoder = RecursivePatternEncoder(config)
        self.hmm = BayesianHMM(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.synthesizer = RecursiveSynthesizer(config)
        self.diffuser = DiffuserInterface(config)
        
        # Interaction history (recent only)
        self.history = deque(maxlen=config.HISTORY_LENGTH)
        
        # Try to load state
        self.load_state(config.STATE_FILE)
        
        self.logger.info("RecursiveWatchDog initialized")
    
    def process_user_message(self, message: str) -> Dict[str, Any]:
        """
        Process a message from the user
        
        Args:
            message: User's text message
            
        Returns:
            Response dictionary with various fields
        """
        try:
            start_time = time.time()
            self.logger.info(f"Processing user message: {message[:50]}...")
            
            # Step 1: Encode the message
            encoded_message = self.encoder.encode(message)
            
            # Step 2: Update HMM with encoded message
            self.hmm.forward_pass(encoded_message)
            
            # Step 3: Detect anomalies
            anomaly_status = self.anomaly_detector.detect_anomaly(encoded_message)
            
            # Step 4: Decide on response type
            response_type = self._decide_response_type(message, anomaly_status)
            
            # Step 5: Generate response
            if response_type == "text":
                # Generate text response
                response_content = self.synthesizer.synthesize_text(
                    self.hmm.get_current_state_index(),
                    history=self.history
                )
                image_bytes = None
                
                result = {
                    "type": "text",
                    "content": response_content,
                    "anomaly_status": anomaly_status
                }
                
            elif response_type == "image":
                # Generate image prompt from user message and HMM state
                prompt = self.synthesizer.synthesize_prompt(
                    self.hmm.get_current_state_index(),
                    user_goal=message
                )
                
                # Generate image
                image_bytes = self.diffuser.generate_image(prompt)
                
                if image_bytes:
                    # Get image description
                    image_stats = self.diffuser.interpret_image(image_bytes)
                    description = self.synthesizer.synthesize_image_description(
                        self.hmm.get_current_state_index(),
                        image_stats
                    )
                    
                    # Process the diffuser output (update HMM with image features)
                    diffuser_response = self.process_diffuser_output(image_bytes)
                    
                    result = {
                        "type": "image",
                        "content": description,
                        "prompt": prompt,
                        "image_base64": self.diffuser.image_to_base64(image_bytes),
                        "image_stats": image_stats,
                        "anomaly_status": anomaly_status
                    }
                else:
                    # Fallback to text if image generation failed
                    response_content = "I wasn't able to generate an image based on your request. " + \
                                      self.synthesizer.synthesize_text(
                                          self.hmm.get_current_state_index(),
                                          history=self.history
                                      )
                    
                    result = {
                        "type": "text",
                        "content": response_content,
                        "error": "Image generation failed",
                        "anomaly_status": anomaly_status
                    }
            else:
                # Shouldn't happen, but just in case
                result = {
                    "type": "error",
                    "content": "Unknown response type",
                    "anomaly_status": anomaly_status
                }
            
            # Update history
            self.history.append({
                "user_message": message,
                "encoded_message": encoded_message,
                "hmm_state": self.hmm.get_current_state_index(),
                "anomaly_status": anomaly_status,
                "response_type": response_type,
                "timestamp": time.time()
            })
            
            # Record interaction
            self.self_ref_manager.record_interaction(
                message,
                result
            )
            
            # Update self-reference periodically
            self._update_self_reference()
            
            # Periodic state saving
            if np.random.random() < 0.1:  # 10% chance per interaction
                self.save_state(self.config.STATE_FILE)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Processed user message in {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing user message: {e}", exc_info=True)
            return {
                "type": "error",
                "content": "I encountered an error while processing your message.",
                "error": str(e)
            }
    
    def process_diffuser_output(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Process the output from the diffusion model
        
        Args:
            image_bytes: Generated image as bytes
            
        Returns:
            Processing result
        """
        try:
            # Interpret the image
            image_stats = self.diffuser.interpret_image(image_bytes)
            
            # Encode the image statistics
            encoded_stats = self.encoder.encode(image_stats)
            
            # Update HMM with the encoded stats
            self.hmm.forward_pass(encoded_stats)
            
            # Check for anomalies
            anomaly_status = self.anomaly_detector.detect_anomaly(encoded_stats)
            
            if anomaly_status != "Red":
                # If not a severe anomaly, add to normal patterns
                self.anomaly_detector.update_normal_patterns(encoded_stats)
            
            # Get the current HMM state sequence
            # Use last few observations from history plus this one
            recent_encodings = [item["encoded_message"] for item in self.history if "encoded_message" in item][-5:]
            recent_encodings.append(encoded_stats)
            
            if len(recent_encodings) > 1:
                # Perform spiral optimization on the state sequence
                optimized_states = self.hmm.spiral_optimize(recent_encodings)
                
                # Update model based on optimized state sequence
                self.hmm.update_model(recent_encodings)
            
            # Generate description
            description = self.synthesizer.synthesize_image_description(
                self.hmm.get_current_state_index(),
                image_stats
            )
            
            return {
                "description": description,
                "stats": image_stats,
                "anomaly_status": anomaly_status,
                "hmm_state": self.hmm.get_current_state_index(),
                "hmm_entropy": self.hmm.current_entropy
            }
            
        except Exception as e:
            self.logger.error(f"Error processing diffuser output: {e}", exc_info=True)
            return {
                "error": str(e),
                "description": "I had trouble interpreting this image."
            }
    
    def process_feedback(self, feedback: Union[float, str], interaction_index: int = -1) -> bool:
        """
        Process feedback on a recent interaction
        
        Args:
            feedback: Feedback value (positive/negative) or text
            interaction_index: Index in history (-1 = most recent)
            
        Returns:
            Success flag
        """
        try:
            # Convert feedback to a score if it's text
            if isinstance(feedback, str):
                # Simple sentiment analysis
                positive_words = {"good", "great", "excellent", "awesome", "nice", "correct", "yes", "right", "like", "perfect", "better"}
                negative_words = {"bad", "wrong", "incorrect", "no", "not", "dislike", "worse", "terrible", "poor", "awful"}
                
                feedback_lower = feedback.lower()
                
                # Count positive and negative words
                positive_count = sum(1 for word in positive_words if word in feedback_lower)
                negative_count = sum(1 for word in negative_words if word in feedback_lower)
                
                # Determine sentiment
                if positive_count > negative_count:
                    feedback_score = 1.0
                elif negative_count > positive_count:
                    feedback_score = -1.0
                else:
                    feedback_score = 0.0
            else:
                # Normalize numerical feedback to [-1, 1]
                feedback_score = np.clip(float(feedback), -1.0, 1.0)
            
            # Get the corresponding history item
            if not self.history:
                self.logger.warning("No history available for feedback")
                return False
            
            try:
                history_item = list(self.history)[interaction_index]
            except IndexError:
                self.logger.error(f"Invalid history index: {interaction_index}")
                return False
            
            # Get the observation from history
            if "encoded_message" not in history_item:
                self.logger.warning("No encoded message in history item")
                return False
            
            encoded_message = history_item["encoded_message"]
            
            # Update HMM with feedback
            self.hmm.update_model([encoded_message], feedback=feedback_score)
            
            # Update self-reference
            self.self_ref_manager.update_description("latest_feedback", {
                "score": feedback_score,
                "text": feedback if isinstance(feedback, str) else None,
                "timestamp": time.time()
            })
            
            self.logger.info(f"Processed feedback: {feedback_score:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}", exc_info=True)
            return False
    
    def _decide_response_type(self, message: str, anomaly_status: str) -> str:
    """
    Decide whether to respond with text or image
    
    Args:
        message: User's text message
        anomaly_status: Anomaly detection status
        
    Returns:
        Response type string: "text" or "image"
    """
    # If severe anomaly detected, always use text
    if anomaly_status == "Red":
        return "text"
    
    # Check for explicit image requests
    image_request_phrases = [
        "show me", "generate", "create", "draw", "make", "visualize",
        "image", "picture", "photo", "illustration", "artwork", "drawing"
    ]
    
    message_lower = message.lower()
    
    # Check if any image request phrase is in the message
    for phrase in image_request_phrases:
        if phrase in message_lower:
            return "image"
    
    # Use HMM state to determine response type
    hmm_state = self.hmm.get_current_state_index()
    
    # If state corresponds to "creative" theme, favor image
    if hmm_state % 5 == 4:  # creative state
        return "image" if np.random.random() < 0.8 else "text"
    
    # For other states, prefer text but occasionally use image
    probs = {
        0: 0.1,  # neutral state: 10% chance of image
        1: 0.2,  # positive state: 20% chance of image
        2: 0.05,  # negative state: 5% chance of image
        3: 0.3,  # curious state: 30% chance of image
    }
    
    prob = probs.get(hmm_state % 5, 0.1)
    
    # Also consider entropy - higher entropy (uncertainty) favors text responses
    entropy_factor = self.hmm.current_entropy
    prob *= (1.0 - entropy_factor * 0.5)  # Reduce probability based on entropy
    
    return "image" if np.random.random() < prob else "text"

def _update_self_reference(self) -> None:
    """Update the self-reference manager with current system state"""
    # Get HMM self-description
    hmm_state = {
        "current_state_index": self.hmm.get_current_state_index(),
        "state_distribution": self.hmm.get_current_state_distribution().tolist(),
        "entropy": self.hmm.current_entropy
    }
    
    # Update self-reference
    self.self_ref_manager.update_description("hmm_state", hmm_state)
    self.self_ref_manager.update_description("last_update", time.time())
    
    # Periodically save description
    if np.random.random() < 0.2:  # 20% chance
        self.self_ref_manager.save_description()

def save_state(self, filepath: str) -> bool:
    """
    Save the system state to file
    
    Args:
        filepath: Path to save the state file
        
    Returns:
        Success flag
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Dictionary to store component states
        state_dict = {}
        
        # Save HMM to a separate file
        hmm_file = os.path.join(os.path.dirname(filepath), "hmm_state.npz")
        hmm_saved = self.hmm.save_model(hmm_file)
        state_dict["hmm_file"] = hmm_file if hmm_saved else None
        
        # Save anomaly detector state
        anomaly_saved = self.anomaly_detector.save_state()
        
        # Save self-reference
        self.self_ref_manager.save_description()
        
        # Store history
        state_dict["history"] = list(self.history)
        
        # Save state dictionary
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)
        
        self.logger.info(f"Saved system state to {filepath}")
        return True
        
    except Exception as e:
        self.logger.error(f"Error saving system state: {e}", exc_info=True)
        return False

def load_state(self, filepath: str) -> bool:
    """
    Load the system state from file
    
    Args:
        filepath: Path to the state file
        
    Returns:
        Success flag
    """
    try:
        if not os.path.exists(filepath):
            self.logger.info("No saved state file found, using default initialization")
            return False
        
        # Load state dictionary
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        
        # Load HMM from separate file
        hmm_file = state_dict.get("hmm_file")
        if hmm_file and os.path.exists(hmm_file):
            self.hmm.load_model(hmm_file)
        
        # Load history if available
        if "history" in state_dict:
            self.history = deque(state_dict["history"], maxlen=self.config.HISTORY_LENGTH)
        
        self.logger.info(f"Loaded system state from {filepath}")
        return True
        
    except Exception as e:
        self.logger.error(f"Error loading system state: {e}", exc_info=True)
        return False
		
		# =================================================================
# Main Function
# =================================================================

async def main():
    """Main function for command-line execution"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SDR-DBHMM-WD: Spiraling Deep Recursive Deep Bayesian Hidden Markov Model Watchdog")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--no-save", action="store_true", help="Disable state saving on exit")
    parser.add_argument("--interactive", action="store_true", help="Start interactive console")
    args = parser.parse_args()
    
    # Initialize configuration
    Config.initialize(args.config)
    
    # Set up logging
    setup_logging(Config)
    
    # Create watchdog
    watchdog = RecursiveWatchDog(Config)
    
    logger = logging.getLogger(__name__)
    logger.info("SDR-DBHMM-WD system started")
    
    if args.interactive:
        try:
            print("SDR-DBHMM-WD Interactive Console")
            print("-------------------------------")
            print("Type 'exit' or 'quit' to close")
            print("Type 'image <prompt>' to generate an image")
            print("Type 'feedback <score>' to provide feedback on the last response")
            print("-------------------------------")
            
            while True:
                # Get user input
                user_input = input("> ")
                
                # Check for exit command
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                
                # Check for feedback command
                if user_input.lower().startswith("feedback "):
                    try:
                        feedback_str = user_input[len("feedback "):].strip()
                        feedback = float(feedback_str)
                        success = watchdog.process_feedback(feedback)
                        if success:
                            print(f"Feedback processed: {feedback}")
                        else:
                            print("Failed to process feedback")
                    except ValueError:
                        print("Feedback must be a number")
                    continue
                
                # Check for image command
                if user_input.lower().startswith("image "):
                    # Force image generation
                    prompt = user_input[len("image "):].strip()
                    result = watchdog.process_user_message(prompt)
                    
                    # Override response type
                    if result["type"] != "error":
                        result["type"] = "image"
                        
                        # Generate image prompt
                        image_prompt = watchdog.synthesizer.synthesize_prompt(
                            watchdog.hmm.get_current_state_index(),
                            user_goal=prompt
                        )
                        
                        # Generate image
                        image_bytes = watchdog.diffuser.generate_image(image_prompt)
                        
                        if image_bytes:
                            # Get image description
                            image_stats = watchdog.diffuser.interpret_image(image_bytes)
                            description = watchdog.synthesizer.synthesize_image_description(
                                watchdog.hmm.get_current_state_index(),
                                image_stats
                            )
                            
                            result = {
                                "type": "image",
                                "content": description,
                                "prompt": image_prompt,
                                "image_stats": image_stats
                            }
                            
                            # Process the diffuser output
                            watchdog.process_diffuser_output(image_bytes)
                            
                            # Save image
                            import random
                            image_path = f"output_image_{int(time.time())}_{random.randint(1000, 9999)}.png"
                            with open(image_path, 'wb') as f:
                                f.write(image_bytes)
                            
                            print(f"Image saved to: {image_path}")
                            print(f"Description: {description}")
                            print(f"Prompt used: {image_prompt}")
                        else:
                            print("Failed to generate image")
                    else:
                        print(f"Error: {result.get('error', 'Unknown error')}")
                    
                    continue
                
                # Process normal message
                result = watchdog.process_user_message(user_input)
                
                # Display result
                if result["type"] == "text":
                    print(f"Response: {result['content']}")
                elif result["type"] == "image":
                    image_path = f"output_image_{int(time.time())}.png"
                    
                    # Decode and save image
                    try:
                        image_bytes = watchdog.diffuser.base64_to_image(result["image_base64"])
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                        
                        print(f"Image saved to: {image_path}")
                        print(f"Description: {result['content']}")
                        print(f"Prompt used: {result['prompt']}")
                    except:
                        print(f"Failed to save image. Description: {result['content']}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
                
                # Show anomaly status if not normal
                if result.get("anomaly_status", "Green") != "Green":
                    print(f"Anomaly status: {result.get('anomaly_status')}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            # Save state on exit
            if not args.no_save:
                print("Saving system state...")
                watchdog.save_state(Config.STATE_FILE)
                print("State saved")
    
    logger.info("SDR-DBHMM-WD system stopped")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
		
	# =================================================================
# Universal AI Communication System - Main Interface
# =================================================================

class UniversalAICommunicationSystem:
    """
    Main integration class for the Universal AI Communication System.
    
    This class integrates all components into a cohesive system that enables:
    1. Communication between different AI architectures
    2. Translation of representations between modalities
    3. Monitoring and analysis of AI communication patterns
    4. Recursive pattern recognition across different AI languages
    
    The system serves as a practical implementation of the thesis concept
    for a universal AI communication language.
    """
    def __init__(self, config_path: Optional[str] = None):
        # Initialize configuration
        self.config = Config()
        if config_path:
            Config.initialize(config_path)
        else:
            Config.initialize()
        
        # Set up logging
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize all core components
        self.logger.info("Initializing Universal AI Communication System components...")
        
        # Self-reference manager for system introspection
        self.self_ref_manager = SelfReferenceManager(self.config)
        
        # Core components from original system
        self.encoder = RecursivePatternEncoder(self.config)
        self.hmm = BayesianHMM(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.synthesizer = RecursiveSynthesizer(self.config)
        self.diffuser = DiffuserInterface(self.config)
        
        # New components for universal AI communication
        self.translator = AIProtocolTranslator(self.config)
        self.universal_language = UniversalAILanguage(self.config)
        
        # Watchdog for coordinating everything
        self.watchdog = RecursiveWatchDog(self.config)
        
        # Registered AI systems
        self.connected_systems = {}
        
        # Translation history
        self.translation_history = deque(maxlen=100)
        
        # Ready flag
        self.ready = True
        
        self.logger.info("Universal AI Communication System initialized and ready")
    
    def register_ai_system(self, system_id: str, system_type: str, 
                         capabilities: List[str]) -> Dict[str, Any]:
        """
        Register an AI system to participate in universal communication
        
        Args:
            system_id: Unique identifier for the AI system
            system_type: Type of AI system (llm, diffusion, rl, etc.)
            capabilities: List of capabilities the system supports
            
        Returns:
            Registration status
        """
        try:
            # First register with the UAL module
            ual_reg = self.universal_language.register_ai_system(
                system_id, system_type, capabilities
            )
            
            # Create system connection record
            connection = {
                "id": system_id,
                "type": system_type,
                "capabilities": capabilities,
                "ual_id": ual_reg.get("ual_id"),
                "connection_time": time.time(),
                "last_activity": time.time(),
                "messages": {
                    "sent": 0,
                    "received": 0,
                    "translated": 0
                }
            }
            
            # Add to connected systems
            self.connected_systems[system_id] = connection
            
            self.logger.info(f"AI system registered: {system_id} ({system_type})")
            
            return {
                "status": "registered",
                "ual_id": ual_reg.get("ual_id"),
                "system_info": connection
            }
            
        except Exception as e:
            self.logger.error(f"Error registering AI system: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def send_message(self, source_id: str, target_id: str, 
                    content: Any, intent: str = "INFORM") -> Dict[str, Any]:
        """
        Send a message from one AI system to another
        
        Args:
            source_id: Source AI system ID
            target_id: Target AI system ID
            content: Message content
            intent: Message intent
            
        Returns:
            Message delivery status
        """
        try:
            # Check if systems are registered
            if source_id not in self.connected_systems:
                raise ValueError(f"Source system not registered: {source_id}")
            if target_id not in self.connected_systems:
                raise ValueError(f"Target system not registered: {target_id}")
            
            # Get system information
            source_system = self.connected_systems[source_id]
            target_system = self.connected_systems[target_id]
            
            # Update activity timestamps
            source_system["last_activity"] = time.time()
            
            # Create a UAL message
            ual_message = self.universal_language.create_message(
                content=content,
                intent=intent,
                sender=source_id,
                receiver=target_id,
                timestamp=time.time()
            )
            
            # Check for anomalies
            message_vector = self.encoder.encode(ual_message)
            anomaly_status = self.anomaly_detector.detect_anomaly(message_vector)
            
            if anomaly_status == "Red":
                self.logger.warning(f"Red anomaly detected in message from {source_id} to {target_id}")
                return {
                    "status": "rejected",
                    "reason": "Anomaly detected",
                    "anomaly_level": "Red"
                }
            
            # Translate for target system
            translated = self.universal_language.translate_between_systems(
                ual_message, source_id, target_id
            )
            
            # Record translation in history
            self.translation_history.append({
                "timestamp": time.time(),
                "source_id": source_id,
                "target_id": target_id,
                "message_id": ual_message.get("header", {}).get("message_id", "unknown"),
                "intent": intent,
                "translated": bool(translated and "error" not in translated)
            })
            
            # Update message counters
            source_system["messages"]["sent"] += 1
            target_system["messages"]["received"] += 1
            target_system["last_activity"] = time.time()
            
            if translated and "error" not in translated:
                source_system["messages"]["translated"] += 1
            
            # Delivery status
            return {
                "status": "delivered",
                "message_id": ual_message.get("header", {}).get("message_id", "unknown"),
                "anomaly_status": anomaly_status,
                "translated": bool(translated and "error" not in translated),
                "target_format": target_system["type"]
            }
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def translate_content(self, content: Any, source_type: str, 
                        target_type: str) -> Dict[str, Any]:
        """
        Translate content between different AI representations
        
        Args:
            content: Content to translate
            source_type: Source representation type
            target_type: Target representation type
            
        Returns:
            Translated content
        """
        try:
            # Use protocol translator for direct translation
            direct_translation = self.translator.translate(
                content, source_type, target_type
            )
            
            # Also use the universal language for comparison
            # First, register temporary systems if needed
            source_id = f"temp_{source_type}_{int(time.time())}"
            target_id = f"temp_{target_type}_{int(time.time())}"
            
            self.universal_language.register_ai_system(
                source_id, source_type, ["translation"]
            )
            
            self.universal_language.register_ai_system(
                target_id, target_type, ["translation"]
            )
            
            # Create UAL message
            ual_message = self.universal_language.create_message(
                content=content,
                intent="INFORM",
                sender=source_id,
                receiver=target_id
            )
            
            # Translate through UAL
            ual_translation = self.universal_language.translate_between_systems(
                ual_message, source_id, target_id
            )
            
            return {
                "direct_translation": direct_translation,
                "ual_translation": ual_translation,
                "source_type": source_type,
                "target_type": target_type
            }
            
        except Exception as e:
            self.logger.error(f"Error translating content: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def analyze_communication(self, system_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze communication patterns for a specific system or all systems
        
        Args:
            system_id: Optional system ID to analyze (all systems if None)
            
        Returns:
            Analysis results
        """
        try:
            results = {
                "timestamp": time.time(),
                "systems": {}
            }
            
            # Filter systems to analyze
            systems_to_analyze = {}
            if system_id:
                if system_id in self.connected_systems:
                    systems_to_analyze[system_id] = self.connected_systems[system_id]
                else:
                    return {"status": "error", "error": f"System not found: {system_id}"}
            else:
                systems_to_analyze = self.connected_systems
            
            # Calculate basic statistics
            total_messages = 0
            total_translations = 0
            system_types = set()
            
            # Analyze each system
            for sys_id, system in systems_to_analyze.items():
                system_types.add(system["type"])
                
                msgs = system["messages"]
                total_sent = msgs.get("sent", 0)
                total_received = msgs.get("received", 0)
                total_translated = msgs.get("translated", 0)
                
                total_messages += total_sent + total_received
                total_translations += total_translated
                
                # Calculate activity level
                now = time.time()
                last_activity = system.get("last_activity", 0)
                hours_since_activity = (now - last_activity) / 3600
                
                if hours_since_activity < 1:
                    activity_level = "high"
                elif hours_since_activity < 24:
                    activity_level = "medium"
                else:
                    activity_level = "low"
                
                # System-specific analysis
                results["systems"][sys_id] = {
                    "type": system["type"],
                    "messages_sent": total_sent,
                    "messages_received": total_received,
                    "translations": total_translated,
                    "activity_level": activity_level,
                    "hours_since_activity": round(hours_since_activity, 2),
                    "capabilities": system.get("capabilities", [])
                }
            
            # Overall statistics
            results["total_messages"] = total_messages
            results["total_translations"] = total_translations
            results["system_types"] = list(system_types)
            results["system_count"] = len(systems_to_analyze)
            
            # Add HMM state information
            results["hmm_state"] = {
                "current_state": self.hmm.get_current_state_index(),
                "entropy": self.hmm.current_entropy
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing communication: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def demonstrate(self) -> Dict[str, Any]:
        """
        Run a complete demonstration of the system capabilities
        
        Returns:
            Demonstration results
        """
        try:
            results = {
                "timestamp": time.time(),
                "components": {},
                "translations": {},
                "analysis": {}
            }
            
            # Step 1: Register sample systems
            system_types = ["llm", "diffusion", "rl", "recommender"]
            for system_type in system_types:
                system_id = f"demo_{system_type}"
                reg_result = self.register_ai_system(
                    system_id, 
                    system_type, 
                    ["text", "universal", system_type]
                )
                results["components"][system_id] = {
                    "registration": reg_result.get("status"),
                    "type": system_type
                }
            
            # Step 2: Test messages between different systems
            test_cases = [
                ("demo_llm", "demo_diffusion", "A beautiful sunset over mountains"),
                ("demo_diffusion", "demo_llm", {"image_type": "landscape", "description": "Ocean waves crashing on rocky shore"}),
                ("demo_llm", "demo_rl", "Move three steps forward then turn right"),
                ("demo_rl", "demo_llm", {"action": "pick_up", "object": "book", "confidence": 0.9}),
                ("demo_recommender", "demo_llm", {"items": ["movie_123", "book_456"], "scores": [0.95, 0.87]}),
                ("demo_llm", "demo_recommender", "I prefer science fiction and mystery genres")
            ]
            
            for source_id, target_id, content in test_cases:
                result = self.send_message(source_id, target_id, content)
                case_key = f"{source_id}_to_{target_id}"
                results["translations"][case_key] = {
                    "content": str(content)[:50] + ("..." if len(str(content)) > 50 else ""),
                    "status": result.get("status"),
                    "translated": result.get("translated", False)
                }
            
            # Step 3: Generate analysis
            results["analysis"] = self.analyze_communication()
            
            # Step 4: Demonstrate UAL capabilities
            ual_demo = self.universal_language.demonstrate_translation()
            results["ual_demonstration"] = {
                "status": ual_demo.get("status"),
                "translation_count": len(ual_demo.get("demo_results", {})),
                "supported_concepts": len(self.universal_language.list_concepts())
            }
            
            # Step 5: Show state information
            hmm_state = self.hmm.get_current_state_index()
            hmm_entropy = self.hmm.current_entropy
            
            results["state_info"] = {
                "hmm_state": hmm_state,
                "hmm_entropy": hmm_entropy,
                "anomaly_status": "Green",  # Assuming normal operation for demo
                "system_uptime": time.time() - results["timestamp"]
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in demonstration: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def save_state(self, filepath: Optional[str] = None) -> bool:
        """
        Save the current system state
        
        Args:
            filepath: Optional filepath (uses default if None)
            
        Returns:
            Success flag
        """
        try:
            if filepath is None:
                filepath = self.config.STATE_FILE
            
            # Use the RecursiveWatchDog to save core components state
            watchdog_saved = self.watchdog.save_state(filepath)
            
            # Create a state dictionary for additional components
            state_dict = {
                "connected_systems": self.connected_systems,
                "translation_history": list(self.translation_history),
                "timestamp": time.time(),
                "version": self.universal_language.get_version()
            }
            
            # Save additional state
            additional_state_file = f"{os.path.splitext(filepath)[0]}_additional.pkl"
            with open(additional_state_file, 'wb') as f:
                pickle.dump(state_dict, f)
            
            # Save self-reference
            self.self_ref_manager.save_description()
            
            self.logger.info(f"System state saved to {filepath} and {additional_state_file}")
            return watchdog_saved
            
        except Exception as e:
            self.logger.error(f"Error saving system state: {e}", exc_info=True)
            return False
    
    def load_state(self, filepath: Optional[str] = None) -> bool:
        """
        Load system state from file
        
        Args:
            filepath: Optional filepath (uses default if None)
            
        Returns:
            Success flag
        """
        try:
            if filepath is None:
                filepath = self.config.STATE_FILE
            
            # Use the RecursiveWatchDog to load core components state
            watchdog_loaded = self.watchdog.load_state(filepath)
            
            # Try to load additional state
            additional_state_file = f"{os.path.splitext(filepath)[0]}_additional.pkl"
            if os.path.exists(additional_state_file):
                with open(additional_state_file, 'rb') as f:
                    state_dict = pickle.load(f)
                
                self.connected_systems = state_dict.get("connected_systems", {})
                self.translation_history = deque(
                    state_dict.get("translation_history", []),
                    maxlen=100
                )
                
                self.logger.info(f"Additional system state loaded from {additional_state_file}")
            
            self.logger.info(f"System state loaded from {filepath}")
            return watchdog_loaded
            
        except Exception as e:
            self.logger.error(f"Error loading system state: {e}", exc_info=True)
            return False

			# =================================================================
# Command Line Interface
# =================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Universal AI Communication System"
    )
    parser.add_argument(
        "--config", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run demonstration mode"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Start interactive console"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save state on exit"
    )
    return parser.parse_args()

def interactive_console(system: UniversalAICommunicationSystem):
    """Run an interactive console for the system"""
    print("Universal AI Communication System - Interactive Console")
    print("-----------------------------------------------------")
    print("Commands:")
    print("  help                 - Show this help")
    print("  exit, quit           - Exit the console")
    print("  demo                 - Run full demonstration")
    print("  register TYPE ID     - Register an AI system")
    print("  send SOURCE TARGET   - Send a message")
    print("  translate SRC TGT    - Translate between types")
    print("  analyze [ID]         - Analyze communication")
    print("  save [PATH]          - Save system state")
    print("  load [PATH]          - Load system state")
    print("  <any text>           - Process as user message")
    print("-----------------------------------------------------")
    
    while True:
        try:
            command = input("> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ["exit", "quit", "q"]:
                break
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == "help":
                print("Universal AI Communication System - Interactive Console")
                print("-----------------------------------------------------")
                print("Commands:")
                print("  help                 - Show this help")
                print("  exit, quit           - Exit the console")
                print("  demo                 - Run full demonstration")
                print("  register TYPE ID     - Register an AI system")
                print("  send SOURCE TARGET   - Send a message")
                print("  translate SRC TGT    - Translate between types")
                print("  analyze [ID]         - Analyze communication")
                print("  save [PATH]          - Save system state")
                print("  load [PATH]          - Load system state")
                print("  <any text>           - Process as user message")
                print("-----------------------------------------------------")
            
            elif cmd == "demo":
                print("Running demonstration...")
                results = system.demonstrate()
                print(f"Demonstration completed with {len(results.get('translations', {}))} translations")
                print(f"Systems: {', '.join(results.get('components', {}).keys())}")
                print(f"HMM State: {results.get('state_info', {}).get('hmm_state')}")
                print(f"Status: {results.get('status', 'success')}")
            
            elif cmd == "register":
                if len(parts) < 3:
                    print("Usage: register TYPE ID [CAPABILITIES]")
                    continue
                
                system_type = parts[1]
                system_id = parts[2]
                capabilities = parts[3:] if len(parts) > 3 else ["universal"]
                
                print(f"Registering {system_type} system with ID {system_id}...")
                result = system.register_ai_system(system_id, system_type, capabilities)
                print(f"Registration status: {result.get('status')}")
                if "error" in result:
                    print(f"Error: {result['error']}")
            
            elif cmd == "send":
                if len(parts) < 3:
                    print("Usage: send SOURCE TARGET CONTENT")
                    continue
                
                source_id = parts[1]
                target_id = parts[2]
                content = " ".join(parts[3:]) if len(parts) > 3 else "Test message"
                
                print(f"Sending message from {source_id} to {target_id}...")
                result = system.send_message(source_id, target_id, content)
                print(f"Delivery status: {result.get('status')}")
                print(f"Translated: {result.get('translated', False)}")
                if "error" in result:
                    print(f"Error: {result['error']}")
            
            elif cmd == "translate":
                if len(parts) < 3:
                    print("Usage: translate SOURCE_TYPE TARGET_TYPE CONTENT")
                    continue
                
                source_type = parts[1]
                target_type = parts[2]
                content = " ".join(parts[3:]) if len(parts) > 3 else "Test content"
                
                print(f"Translating from {source_type} to {target_type}...")
                result = system.translate_content(content, source_type, target_type)
                print(f"Direct translation: {result.get('direct_translation')}")
                print(f"UAL translation: {result.get('ual_translation')}")
                if "error" in result:
                    print(f"Error: {result['error']}")
            
            elif cmd == "analyze":
                system_id = parts[1] if len(parts) > 1 else None
                target = system_id if system_id else "all systems"
                
                print(f"Analyzing communication for {target}...")
                result = system.analyze_communication(system_id)
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"Total messages: {result.get('total_messages', 0)}")
                    print(f"Total translations: {result.get('total_translations', 0)}")
                    print(f"HMM state: {result.get('hmm_state', {}).get('current_state')}")
                    print(f"HMM entropy: {result.get('hmm_state', {}).get('entropy', 0):.2f}")
                    print(f"Systems analyzed: {len(result.get('systems', {}))}")
            
            elif cmd == "save":
                filepath = parts[1] if len(parts) > 1 else None
                path_info = filepath if filepath else "default path"
                
                print(f"Saving system state to {path_info}...")
                success = system.save_state(filepath)
                print(f"Save {'successful' if success else 'failed'}")
            
            elif cmd == "load":
                filepath = parts[1] if len(parts) > 1 else None
                path_info = filepath if filepath else "default path"
                
                print(f"Loading system state from {path_info}...")
                success = system.load_state(filepath)
                print(f"Load {'successful' if success else 'failed'}")
            
            else:
                # Treat unknown command as user message
                print(f"Processing your message: {command}")
                result = system.watchdog.process_user_message(command)
                
                # Display result
                if result["type"] == "text":
                    print(f"Response: {result['content']}")
                elif result["type"] == "image":
                    print(f"Generated image based on prompt: {result['prompt']}")
                    print(f"Description: {result['content']}")
                    print(f"Image saved to: output_image_{int(time.time())}.png")
                else:
                    print(f"Response type: {result['type']}")
        
        except KeyboardInterrupt:
            print("\nOperation interrupted")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Create the system
    system = UniversalAICommunicationSystem(args.config)
    
    try:
        # Run in demo mode if requested
        if args.demo:
            print("Running demonstration...")
            results = system.demonstrate()
            print(json.dumps(results, indent=2))
        
        # Run interactive console if requested
        elif args.interactive:
            interactive_console(system)
        
        # Otherwise just exit
        else:
            print("System initialized. Use --demo or --interactive for more options.")
    
    finally:
        # Save state unless disabled
        if not args.no_save:
            print("Saving system state...")
            system.save_state()

if __name__ == "__main__":
    main()
	
	
