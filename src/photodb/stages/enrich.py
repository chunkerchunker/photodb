from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MetadataEnricher:
    """Enrich metadata with derived information."""
    
    @staticmethod
    def enrich_location(latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Enrich location data with reverse geocoding.
        
        Args:
            latitude: GPS latitude
            longitude: GPS longitude
            
        Returns:
            Dict with location details
        """
        # This would integrate with a geocoding service
        # For now, return basic info
        return {
            'coordinates': {
                'latitude': latitude,
                'longitude': longitude
            },
            'map_url': f"https://maps.google.com/?q={latitude},{longitude}"
        }
    
    @staticmethod
    def enrich_datetime(captured_at: datetime) -> Dict[str, Any]:
        """
        Enrich datetime with additional temporal information.
        
        Args:
            captured_at: Photo capture datetime
            
        Returns:
            Dict with temporal details
        """
        return {
            'datetime': captured_at.isoformat(),
            'date': captured_at.date().isoformat(),
            'time': captured_at.time().isoformat(),
            'year': captured_at.year,
            'month': captured_at.month,
            'day': captured_at.day,
            'weekday': captured_at.strftime('%A'),
            'hour': captured_at.hour,
            'season': get_season(captured_at),
            'time_of_day': get_time_of_day(captured_at.hour)
        }
    
    @staticmethod
    def analyze_shooting_conditions(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze shooting conditions from metadata.
        
        Args:
            metadata: Extracted metadata
            
        Returns:
            Dict with shooting analysis
        """
        conditions = {}
        
        # Analyze ISO
        if 'iso' in metadata:
            iso = metadata['iso']
            if iso <= 200:
                conditions['light_conditions'] = 'bright'
            elif iso <= 800:
                conditions['light_conditions'] = 'normal'
            elif iso <= 3200:
                conditions['light_conditions'] = 'dim'
            else:
                conditions['light_conditions'] = 'dark'
        
        # Analyze aperture
        if 'f_number' in metadata:
            f = metadata['f_number']
            if f <= 2.8:
                conditions['depth_of_field'] = 'shallow'
            elif f <= 5.6:
                conditions['depth_of_field'] = 'moderate'
            else:
                conditions['depth_of_field'] = 'deep'
        
        # Analyze shutter speed
        if 'exposure_time' in metadata:
            exp = metadata['exposure_time']
            if isinstance(exp, (int, float)):
                if exp < 1/500:
                    conditions['motion'] = 'frozen'
                elif exp < 1/60:
                    conditions['motion'] = 'normal'
                else:
                    conditions['motion'] = 'motion_blur'
        
        # Flash usage
        if 'flash' in metadata:
            conditions['flash_used'] = metadata['flash'] != 0
        
        return conditions

def get_season(dt: datetime) -> str:
    """Get season from datetime (Northern Hemisphere)."""
    month = dt.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

def get_time_of_day(hour: int) -> str:
    """Get time of day from hour."""
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'