"""Sync state management for incremental data ingestion.

Tracks ingestion progress using cursor-based or timestamp-based checkpointing
to enable idempotent and resumable data harvesting.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class SyncState:
    """State for an incremental sync operation."""
    survey: str
    last_sync_timestamp: Optional[datetime] = None
    cursor: Optional[str] = None
    records_processed: int = 0
    last_successful_id: Optional[str] = None
    error_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SyncStateManager:
    """Manager for persisting and retrieving sync state."""
    
    def __init__(self, state_dir: str = "data/sync_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, Lock] = {}
    
    def _get_lock(self, survey: str) -> Lock:
        """Get thread-safe lock for a survey."""
        if survey not in self._locks:
            self._locks[survey] = Lock()
        return self._locks[survey]
    
    def _state_file(self, survey: str) -> Path:
        """Get state file path for a survey."""
        return self.state_dir / f"{survey.lower()}_state.json"
    
    def load_state(self, survey: str) -> Optional[SyncState]:
        """Load sync state for a survey.
        
        Args:
            survey: Survey name (e.g., 'SDSS', 'Gaia')
        
        Returns:
            SyncState object or None if no state exists
        """
        state_file = self._state_file(survey)
        
        with self._get_lock(survey):
            if not state_file.exists():
                logger.info(f"No sync state found for {survey}")
                return None
            
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                # Convert timestamp string back to datetime
                if data.get('last_sync_timestamp'):
                    data['last_sync_timestamp'] = datetime.fromisoformat(
                        data['last_sync_timestamp']
                    )
                
                state = SyncState(**data)
                logger.info(f"Loaded state for {survey}: {state.records_processed} records")
                return state
            
            except Exception as e:
                logger.error(f"Failed to load state for {survey}: {e}")
                return None
    
    def save_state(self, state: SyncState) -> bool:
        """Persist sync state atomically.
        
        Args:
            state: SyncState to save
        
        Returns:
            True if successful, False otherwise
        """
        state_file = self._state_file(state.survey)
        temp_file = state_file.with_suffix('.tmp')
        
        with self._get_lock(state.survey):
            try:
                # Convert to dict and handle datetime serialization
                data = asdict(state)
                if data.get('last_sync_timestamp'):
                    data['last_sync_timestamp'] = data['last_sync_timestamp'].isoformat()
                
                # Write to temp file first
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Atomic rename
                temp_file.replace(state_file)
                logger.info(f"Saved state for {state.survey}: {state.records_processed} records")
                return True
            
            except Exception as e:
                logger.error(f"Failed to save state for {state.survey}: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                return False
    
    def update_state(
        self,
        survey: str,
        records_processed: Optional[int] = None,
        cursor: Optional[str] = None,
        last_successful_id: Optional[str] = None,
        increment_errors: bool = False
    ) -> bool:
        """Update specific fields in sync state.
        
        Args:
            survey: Survey name
            records_processed: Total records processed (sets absolute value)
            cursor: New cursor position
            last_successful_id: Last successfully processed ID
            increment_errors: Whether to increment error count
        
        Returns:
            True if successful
        """
        state = self.load_state(survey) or SyncState(survey=survey)
        
        if records_processed is not None:
            state.records_processed = records_processed
        if cursor is not None:
            state.cursor = cursor
        if last_successful_id is not None:
            state.last_successful_id = last_successful_id
        if increment_errors:
            state.error_count += 1
        
        state.last_sync_timestamp = datetime.now()
        
        return self.save_state(state)
    
    def reset_state(self, survey: str) -> bool:
        """Reset sync state for a survey.
        
        Args:
            survey: Survey name
        
        Returns:
            True if successful
        """
        state_file = self._state_file(survey)
        
        with self._get_lock(survey):
            try:
                if state_file.exists():
                    state_file.unlink()
                logger.info(f"Reset state for {survey}")
                return True
            except Exception as e:
                logger.error(f"Failed to reset state for {survey}: {e}")
                return False
    
    def get_all_states(self) -> Dict[str, SyncState]:
        """Get sync states for all surveys.
        
        Returns:
            Dictionary mapping survey name to SyncState
        """
        states = {}
        for state_file in self.state_dir.glob("*_state.json"):
            survey = state_file.stem.replace('_state', '').upper()
            state = self.load_state(survey)
            if state:
                states[survey] = state
        return states


# Global singleton instance
_manager: Optional[SyncStateManager] = None


def get_sync_manager(state_dir: str = "data/sync_state") -> SyncStateManager:
    """Get or create global sync state manager.
    
    Args:
        state_dir: Directory for state files
    
    Returns:
        SyncStateManager instance
    """
    global _manager
    if _manager is None:
        _manager = SyncStateManager(state_dir)
    return _manager
