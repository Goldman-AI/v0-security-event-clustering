"""
Security Event Parser Module
Handles parsing of semi-structured security events in key=value format
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


@dataclass
class SecurityEvent:
    """Represents a parsed security event"""
    timestamp: Optional[datetime] = None
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    source_port: Optional[int] = None
    dest_port: Optional[int] = None
    subsystem: Optional[str] = None
    user: Optional[str] = None
    action: Optional[str] = None
    content: Optional[str] = None
    severity: Optional[str] = None
    protocol: Optional[str] = None
    raw_fields: Dict[str, Any] = field(default_factory=dict)


class SecurityEventParser:
    """Parser for semi-structured security events in key=value format"""
    
    # Pattern to match key=value pairs, handling quoted values
    KV_PATTERN = re.compile(
        r"(\w+)=(?:'([^']*)'|\"([^\"]*)\"|(\S+))"
    )
    
    # Common field name mappings (normalize different naming conventions)
    FIELD_MAPPINGS = {
        'timestamp': ['timestamp', 'time', 'datetime', 'date', 'eventtime', 'logtime'],
        'source_ip': ['sourceip', 'srcip', 'src_ip', 'source', 'sip', 'src'],
        'dest_ip': ['destip', 'dstip', 'dst_ip', 'destination', 'dip', 'dst'],
        'source_port': ['sourceport', 'srcport', 'src_port', 'sport'],
        'dest_port': ['destport', 'dstport', 'dst_port', 'dport'],
        'subsystem': ['subsys', 'subsystem', 'module', 'component', 'type', 'logtype'],
        'user': ['user', 'username', 'usr', 'account', 'userid'],
        'action': ['action', 'act', 'event', 'eventtype', 'operation'],
        'content': ['content', 'message', 'msg', 'description', 'desc', 'reason'],
        'severity': ['severity', 'level', 'priority', 'sev', 'loglevel'],
        'protocol': ['protocol', 'proto', 'prot', 'service'],
    }
    
    # Reverse mapping for quick lookup
    _reverse_mapping: Dict[str, str] = {}
    
    def __init__(self):
        # Build reverse mapping
        for standard_name, variants in self.FIELD_MAPPINGS.items():
            for variant in variants:
                self._reverse_mapping[variant.lower()] = standard_name
    
    def parse_line(self, line: str) -> SecurityEvent:
        """Parse a single log line into a SecurityEvent"""
        event = SecurityEvent()
        raw_fields = {}
        
        # Find all key=value pairs
        for match in self.KV_PATTERN.finditer(line):
            key = match.group(1).lower()
            # Get value from whichever group matched (quoted or unquoted)
            value = match.group(2) or match.group(3) or match.group(4)
            
            raw_fields[key] = value
            
            # Map to standard field if recognized
            standard_name = self._reverse_mapping.get(key)
            if standard_name:
                self._set_field(event, standard_name, value)
        
        event.raw_fields = raw_fields
        return event
    
    def _set_field(self, event: SecurityEvent, field_name: str, value: str):
        """Set a field on the event with appropriate type conversion"""
        try:
            if field_name == 'timestamp':
                # Try multiple timestamp formats
                for fmt in [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y/%m/%d %H:%M:%S',
                    '%d/%m/%Y %H:%M:%S',
                    '%Y-%m-%d %H:%M:%S.%f',
                ]:
                    try:
                        event.timestamp = datetime.strptime(value, fmt)
                        break
                    except ValueError:
                        continue
            elif field_name in ['source_port', 'dest_port']:
                setattr(event, field_name, int(value))
            else:
                setattr(event, field_name, value)
        except (ValueError, TypeError):
            # Store as raw if conversion fails
            pass
    
    def parse_file(self, file_path: str, max_lines: Optional[int] = None) -> List[SecurityEvent]:
        """Parse a file of security events"""
        events = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if line:
                    events.append(self.parse_line(line))
        return events
    
    def parse_lines(self, lines: List[str]) -> List[SecurityEvent]:
        """Parse a list of log lines"""
        return [self.parse_line(line.strip()) for line in lines if line.strip()]
    
    def events_to_dataframe(self, events: List[SecurityEvent]) -> pd.DataFrame:
        """Convert list of events to pandas DataFrame"""
        records = []
        for event in events:
            record = {
                'timestamp': event.timestamp,
                'source_ip': event.source_ip,
                'dest_ip': event.dest_ip,
                'source_port': event.source_port,
                'dest_port': event.dest_port,
                'subsystem': event.subsystem,
                'user': event.user,
                'action': event.action,
                'content': event.content,
                'severity': event.severity,
                'protocol': event.protocol,
            }
            # Add any extra fields from raw_fields
            for key, value in event.raw_fields.items():
                if key not in self._reverse_mapping:
                    record[f'extra_{key}'] = value
            records.append(record)
        
        return pd.DataFrame(records)


def extract_ip_features(ip: str) -> Dict[str, int]:
    """Extract numerical features from IP address"""
    if not ip:
        return {'octet_1': 0, 'octet_2': 0, 'octet_3': 0, 'octet_4': 0}
    
    try:
        octets = ip.split('.')
        return {
            'octet_1': int(octets[0]),
            'octet_2': int(octets[1]),
            'octet_3': int(octets[2]),
            'octet_4': int(octets[3]),
        }
    except (IndexError, ValueError):
        return {'octet_1': 0, 'octet_2': 0, 'octet_3': 0, 'octet_4': 0}


if __name__ == "__main__":
    # Test parsing
    test_lines = [
        "timestamp=2020-02-02 01:00:00 sourceip=22.2.2.2 destip=3.4.3.2 destport=4444 subsys=vpn user=a content='blocked by user policy'",
        "timestamp=2020-02-02 01:05:00 srcip=192.168.1.1 dstip=10.0.0.1 dport=443 subsystem=firewall action=allow msg='connection established'",
        "time=2020-02-02 01:10:00 source=8.8.8.8 destination=192.168.1.100 destport=80 type=waf severity=high message='SQL injection attempt detected'",
    ]
    
    parser = SecurityEventParser()
    for line in test_lines:
        event = parser.parse_line(line)
        print(f"Subsystem: {event.subsystem}, Source: {event.source_ip}, Dest: {event.dest_ip}, Content: {event.content}")
