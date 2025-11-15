import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL and detect environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL environment variable is not set!")

IS_CLOUD_RUN = os.getenv("K_SERVICE") is not None  # Cloud Run sets this
IS_NEON = "neon.tech" in DATABASE_URL

# Ensure PostgreSQL SSL mode is enabled (required by Neon)
if "postgresql://" in DATABASE_URL or "postgres://" in DATABASE_URL:
    # Convert postgres:// to postgresql:// (some libraries need this)
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    # Add SSL mode if not present for Neon
    if IS_NEON and "sslmode" not in DATABASE_URL:
        separator = "&" if "?" in DATABASE_URL else "?"
        DATABASE_URL += f"{separator}sslmode=require"

# Log connection (hide password for security)
safe_url = DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'localhost'
logger.info(f"📊 Connecting to PostgreSQL: {safe_url}")
logger.info(f"🔧 Environment: {'Cloud Run' if IS_CLOUD_RUN else 'Neon' if IS_NEON else 'Local Docker'}")

# Choose pool configuration based on environment
if IS_CLOUD_RUN or IS_NEON:
    # Serverless/Neon: Use NullPool (no persistent connections)
    pool_config = {
        "poolclass": NullPool,
    }
    logger.info("📦 Using NullPool (serverless optimized)")
else:
    # Local Docker: Use default QueuedPool with connection pooling
    pool_config = {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 3600,  # Recycle connections after 1 hour
    }
    logger.info("📦 Using QueuedPool (Docker optimized)")

# Connection arguments based on environment
connect_args = {
    "connect_timeout": 10,
}

if IS_NEON:
    connect_args.update({
        "sslmode": "require",           # Neon requires SSL
        "keepalives": 1,                 # Enable TCP keepalives
        "keepalives_idle": 30,           # Start keepalives after 30 seconds
        "keepalives_interval": 10,       # Send keepalive every 10 seconds
        "keepalives_count": 5,           # Close after 5 failed keepalives
        "application_name": "krishisetu_api"  # Identify your app in Neon dashboard
    })
else:
    connect_args["sslmode"] = "prefer"  # Local doesn't require SSL

# Create engine with optimized configuration
engine = create_engine(
    DATABASE_URL,
    **pool_config,
    connect_args=connect_args,
    pool_pre_ping=True,  # Verify connection health before using
    echo=False,  # Set to True for SQL debugging
    execution_options={
        "postgresql_readonly": False,
        "postgresql_insert_executemany_returning": True,
    }
)

# Test connection on startup
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]
        logger.info(f"✅ Connected successfully!")
        logger.info(f"📌 Database version: {version[:50]}...")
except Exception as e:
    logger.error(f"❌ Failed to connect to database: {e}")
    raise

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Prevent issues with detached instances
)

# Base class for all models
Base = declarative_base()

# Dependency for FastAPI endpoints
def get_db():
    """
    Database session dependency for FastAPI.
    Usage: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Event listener: Log slow queries (optional, for debugging)
@event.listens_for(engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
    conn.info.setdefault('query_start_time', []).append(
        __import__('time').time()
    )

@event.listens_for(engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, params, context, executemany):
    total = __import__('time').time() - conn.info['query_start_time'].pop()
    if total > 1.0:  # Log queries taking more than 1 second
        logger.warning(f"⚠️ Slow query ({total:.2f}s): {statement[:100]}...")

# Initialize database tables
def init_db():
    """
    Create all tables in the database.
    Call this after importing all models.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified successfully")
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        raise

# Health check function
def check_db_health():
    """
    Check if database connection is healthy.
    Returns: tuple (bool, str) - (is_healthy, message)
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Database connection healthy"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"

# Cleanup function for graceful shutdown
def close_db():
    """
    Close database connections gracefully.
    Call this on application shutdown.
    """
    try:
        engine.dispose()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error closing database: {e}")

# Print configuration info
logger.info("=" * 60)
logger.info("DATABASE CONFIGURATION SUMMARY")
logger.info(f"Environment: {'Cloud Run' if IS_CLOUD_RUN else 'Neon' if IS_NEON else 'Local'}")
logger.info(f"SSL Mode: {'Required' if IS_NEON else 'Preferred'}")
logger.info(f"Connection timeout: 10 seconds")
logger.info("=" * 60)