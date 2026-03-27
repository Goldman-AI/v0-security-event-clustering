"""
Synthetic Security Event Data Generator
For testing and demonstration purposes
"""

import random
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np


class SecurityEventGenerator:
    """
    Generates realistic synthetic security events for testing
    """
    
    SUBSYSTEMS = ['firewall', 'ips', 'ddos', 'waf', 'webfilter', 'mail', 'vpn', 'antivirus', 'ids']
    
    SEVERITIES = ['low', 'medium', 'high', 'critical']
    
    ACTIONS = ['allow', 'deny', 'block', 'drop', 'alert', 'quarantine', 'redirect']
    
    PROTOCOLS = ['tcp', 'udp', 'http', 'https', 'ftp', 'ssh', 'smtp', 'dns', 'icmp']
    
    # Common attack patterns
    ATTACK_PATTERNS = {
        'sql_injection': {
            'subsystems': ['waf'],
            'ports': [80, 443, 8080, 8443],
            'contents': [
                "SQL injection attempt detected",
                "Blocked malicious SQL query",
                "XSS attack pattern detected",
                "Union select statement blocked",
            ],
            'severity': ['high', 'critical']
        },
        'bruteforce': {
            'subsystems': ['firewall', 'vpn', 'ids'],
            'ports': [22, 23, 3389, 1194],
            'contents': [
                "Multiple failed login attempts",
                "Bruteforce attack detected",
                "Account lockout triggered",
                "Authentication failure threshold exceeded",
            ],
            'severity': ['medium', 'high']
        },
        'ddos': {
            'subsystems': ['ddos', 'firewall'],
            'ports': [80, 443, 53],
            'contents': [
                "SYN flood detected",
                "UDP flood attack",
                "DDoS mitigation activated",
                "Rate limit exceeded",
                "Connection limit reached",
            ],
            'severity': ['high', 'critical']
        },
        'malware': {
            'subsystems': ['antivirus', 'mail', 'webfilter'],
            'ports': [25, 110, 143, 993, 995],
            'contents': [
                "Malware signature detected",
                "Suspicious file blocked",
                "Trojan detected and quarantined",
                "Phishing attempt blocked",
                "Malicious attachment removed",
            ],
            'severity': ['high', 'critical']
        },
        'normal': {
            'subsystems': ['firewall', 'vpn', 'webfilter'],
            'ports': [80, 443, 22, 53, 123],
            'contents': [
                "Connection established",
                "Session started",
                "Traffic allowed by policy",
                "User authenticated successfully",
                "File download completed",
            ],
            'severity': ['low', 'low', 'low', 'medium']  # Weighted toward low
        },
        'policy_violation': {
            'subsystems': ['webfilter', 'firewall', 'mail'],
            'ports': [80, 443, 8080],
            'contents': [
                "Blocked by user policy",
                "Category blocked: social media",
                "Unauthorized application detected",
                "Policy violation: file sharing",
                "URL blocked: gambling category",
            ],
            'severity': ['low', 'medium']
        },
        'port_scan': {
            'subsystems': ['ips', 'firewall', 'ids'],
            'ports': list(range(1, 1024)),
            'contents': [
                "Port scan detected",
                "Reconnaissance activity",
                "Multiple port connection attempts",
                "Network scanning blocked",
            ],
            'severity': ['medium', 'high']
        },
        'vpn_activity': {
            'subsystems': ['vpn'],
            'ports': [1194, 443, 500, 4500],
            'contents': [
                "VPN tunnel established",
                "VPN connection terminated",
                "VPN authentication successful",
                "VPN tunnel timeout",
                "Invalid VPN credentials",
            ],
            'severity': ['low', 'low', 'medium']
        }
    }
    
    # Common IP ranges (for realistic data)
    INTERNAL_IP_RANGES = [
        ('192.168.', 1, 255),
        ('10.0.', 0, 255),
        ('172.16.', 0, 255),
    ]
    
    EXTERNAL_IP_RANGES = [
        ('8.8.', 4, 8),  # Google DNS-like
        ('1.1.', 1, 1),  # Cloudflare-like
        ('203.', 0, 255),
        ('45.33.', 0, 255),
        ('104.16.', 0, 255),
        ('185.', 0, 255),
    ]
    
    USERS = [
        'admin', 'user1', 'user2', 'guest', 'system', 'backup',
        'webadmin', 'dbadmin', 'netadmin', 'developer', 'analyst'
    ]
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _generate_internal_ip(self) -> str:
        prefix, start, end = random.choice(self.INTERNAL_IP_RANGES)
        return f"{prefix}{random.randint(start, end)}.{random.randint(1, 254)}"
    
    def _generate_external_ip(self) -> str:
        prefix, start, end = random.choice(self.EXTERNAL_IP_RANGES)
        return f"{prefix}{random.randint(start, end)}.{random.randint(1, 254)}"
    
    def _generate_timestamp(self, start_date: datetime, end_date: datetime) -> datetime:
        delta = end_date - start_date
        random_seconds = random.randint(0, int(delta.total_seconds()))
        return start_date + timedelta(seconds=random_seconds)
    
    def generate_event(
        self,
        pattern: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """Generate a single security event"""
        
        # Select pattern
        if pattern is None:
            # Weighted selection (more normal traffic)
            patterns = list(self.ATTACK_PATTERNS.keys())
            weights = [0.4 if p == 'normal' else 0.1 for p in patterns]
            pattern = random.choices(patterns, weights=weights)[0]
        
        config = self.ATTACK_PATTERNS[pattern]
        
        # Generate fields
        if timestamp is None:
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
        
        ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Source/dest IPs (attacks typically external->internal)
        if pattern in ['sql_injection', 'bruteforce', 'ddos', 'port_scan']:
            source_ip = self._generate_external_ip()
            dest_ip = self._generate_internal_ip()
        elif pattern == 'vpn_activity':
            source_ip = self._generate_external_ip()
            dest_ip = self._generate_internal_ip()
        else:
            # Mix of directions
            if random.random() > 0.5:
                source_ip = self._generate_internal_ip()
                dest_ip = self._generate_external_ip()
            else:
                source_ip = self._generate_external_ip()
                dest_ip = self._generate_internal_ip()
        
        subsystem = random.choice(config['subsystems'])
        dest_port = random.choice(config['ports'])
        source_port = random.randint(1024, 65535)
        content = random.choice(config['contents'])
        severity = random.choice(config['severity'])
        action = random.choice(self.ACTIONS)
        protocol = random.choice(self.PROTOCOLS)
        user = random.choice(self.USERS) if random.random() > 0.3 else None
        
        # Build event string
        parts = [
            f"timestamp={ts_str}",
            f"sourceip={source_ip}",
            f"destip={dest_ip}",
            f"sourceport={source_port}",
            f"destport={dest_port}",
            f"subsys={subsystem}",
            f"action={action}",
            f"severity={severity}",
            f"protocol={protocol}",
        ]
        
        if user:
            parts.append(f"user={user}")
        
        parts.append(f"content='{content}'")
        
        return ' '.join(parts)
    
    def generate_dataset(
        self,
        n_events: int = 10000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        pattern_distribution: Optional[dict] = None
    ) -> List[str]:
        """
        Generate a dataset of security events
        
        Args:
            n_events: Number of events to generate
            start_date: Start date for timestamps
            end_date: End date for timestamps
            pattern_distribution: Optional dict mapping pattern -> probability
        
        Returns:
            List of event strings
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        events = []
        patterns = list(self.ATTACK_PATTERNS.keys())
        
        if pattern_distribution:
            weights = [pattern_distribution.get(p, 0.1) for p in patterns]
        else:
            # Default: mostly normal traffic with some attacks
            weights = [
                0.40,  # sql_injection
                0.05,  # bruteforce
                0.05,  # ddos
                0.03,  # malware
                0.25,  # normal
                0.10,  # policy_violation
                0.02,  # port_scan
                0.10,  # vpn_activity
            ]
        
        for _ in range(n_events):
            pattern = random.choices(patterns, weights=weights)[0]
            timestamp = self._generate_timestamp(start_date, end_date)
            event = self.generate_event(pattern=pattern, timestamp=timestamp)
            events.append(event)
        
        return events
    
    def save_dataset(self, events: List[str], file_path: str):
        """Save events to file"""
        with open(file_path, 'w') as f:
            for event in events:
                f.write(event + '\n')
        print(f"Saved {len(events)} events to {file_path}")


if __name__ == "__main__":
    # Generate sample dataset
    generator = SecurityEventGenerator(seed=42)
    
    # Generate 1000 events
    events = generator.generate_dataset(n_events=1000)
    
    # Print some samples
    print("Sample events:")
    print("-" * 80)
    for event in events[:5]:
        print(event)
        print()
