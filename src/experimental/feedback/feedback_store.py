
import json
import logging
from typing import Dict, Any

class FeedbackStore:
    """
    Stores query evaluations and user feedback for future analysis and fine-tuning.
    """
    def __init__(self, feedback_file: str = "feedback_log.jsonl"):
        self.feedback_file = feedback_file
        logging.info(f"FeedbackStore initialized. Storing feedback in: {self.feedback_file}")

    def log_feedback(self, query_id: str, evaluation: Dict[str, Any], user_rating: int = None, user_comment: str = None):
        """
        Logs the evaluation results and any user-provided feedback for a given query.
        """
        feedback_entry = {
            "query_id": query_id,
            "evaluation": evaluation,
            "user_feedback": {
                "rating": user_rating,
                "comment": user_comment
            }
        }
        
        try:
            with open(self.feedback_file, "a") as f:
                f.write(json.dumps(feedback_entry) + "\n")
            logging.info(f"Successfully logged feedback for query_id: {query_id}")
        except IOError as e:
            logging.error(f"Failed to log feedback for query_id: {query_id}. Error: {e}")

    # def update_feedback(self, query_id: str, user_rating: int, user_comment: str | None):
    #     """
    #     Finds a specific query log by its ID and updates it with user feedback.
    #     This is not efficient for large files, as it reads and rewrites the entire file.
    #     For a production system, a proper database would be used.
    #     """
    #     try:
    #         with open(self.feedback_file, "r") as f:
    #             lines = f.readlines()

    #         updated_lines = []
    #         found = False
    #         for line in lines:
    #             entry = json.loads(line)
    #             if entry.get("query_id") == query_id:
    #                 entry["user_feedback"]["rating"] = user_rating
    #                 entry["user_feedback"]["comment"] = user_comment
    #                 updated_lines.append(json.dumps(entry) + "\n")
    #                 found = True
    #             else:
    #                 updated_lines.append(line)
            
    #         if not found:
    #             raise ValueError(f"Query ID '{query_id}' not found in feedback log.")

    #         with open(self.feedback_file, "w") as f:
    #             f.writelines(updated_lines)
            
    #         logging.info(f"Successfully updated feedback for query_id: {query_id}")

    #     except FileNotFoundError:
    #         logging.error(f"Feedback file not found at {self.feedback_file}. Cannot update feedback.")
    #         raise
    #     except (IOError, json.JSONDecodeError) as e:
    #         logging.error(f"Error reading or writing feedback file for query_id: {query_id}. Error: {e}")
    #         raise