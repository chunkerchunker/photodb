#!/usr/bin/env python3
import os
import psutil
import subprocess

def check_file_descriptors():
    """Check current file descriptor usage."""
    try:
        # Get current process
        pid = os.getpid()
        process = psutil.Process(pid)
        
        # Count open file descriptors
        open_files = len(process.open_files())
        print(f"Open file descriptors: {open_files}")
        
        # Check system limits
        soft_limit, hard_limit = process.rlimit(psutil.RLIMIT_NOFILE)
        print(f"File descriptor limits: {soft_limit} (soft) / {hard_limit} (hard)")
        
        # Show detailed file info
        if open_files > 0:
            print("\nOpen files:")
            for f in process.open_files():
                print(f"  {f.path} (fd: {f.fd})")
                
    except Exception as e:
        print(f"Error checking file descriptors: {e}")

def check_postgres_connections():
    """Check PostgreSQL connection count."""
    try:
        result = subprocess.run([
            "psql", "-d", "photodb", "-t", "-c", 
            "SELECT count(*) FROM pg_stat_activity WHERE datname = 'photodb';"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            count = result.stdout.strip()
            print(f"PostgreSQL connections to photodb: {count}")
        else:
            print("Could not check PostgreSQL connections")
    except Exception as e:
        print(f"Error checking PostgreSQL: {e}")

if __name__ == "__main__":
    print("=== Resource Usage Check ===")
    check_file_descriptors()
    print()
    check_postgres_connections()