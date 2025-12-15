# Full path: backend/scripts/reset_knowledge_base.py
"""
Knowledge Base Reset Script

Purges legacy verification data from the database to ensure clean RAG retrieval.
Use this after upgrading to the new Tavily-filtered pipeline to remove "poisoned" data.

SAFETY: Requires explicit confirmation or --force flag.
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.core.db import get_sessionmaker
from app.core.logging import get_logger

logger = get_logger(__name__)


async def reset_knowledge_base(*, dry_run: bool = True):
    """Reset the knowledge base by truncating claims and evidence tables.
    
    Args:
        dry_run: If True, only show what would be deleted without executing.
    """
    session_maker = get_sessionmaker()
    
    async with session_maker() as session:
        # Count existing records
        claims_count_result = await session.execute(text("SELECT COUNT(*) FROM claims"))
        claims_count = claims_count_result.scalar()
        
        evidence_count_result = await session.execute(text("SELECT COUNT(*) FROM evidence"))
        evidence_count = evidence_count_result.scalar()
        
        sources_count_result = await session.execute(text("SELECT COUNT(*) FROM sources"))
        sources_count = sources_count_result.scalar()
        
        print(f"\n📊 Current Knowledge Base Status:")
        print(f"   Claims:   {claims_count:,}")
        print(f"   Evidence: {evidence_count:,}")
        print(f"   Sources:  {sources_count:,}")
        
        if dry_run:
            print(f"\n🔍 DRY RUN MODE - No changes will be made")
            print(f"\n⚠️  To execute the reset, run with --force flag:")
            print(f"   python backend/scripts/reset_knowledge_base.py --force")
            return
        
        print(f"\n⚠️  WARNING: This will permanently delete all claims and evidence!")
        print(f"   Verifications table will be preserved (for audit trail).")
        print(f"   Sources table will be preserved (for reference).")
        
        # Execute truncation
        try:
            print(f"\n🗑️  Truncating evidence table...")
            await session.execute(text("TRUNCATE TABLE evidence CASCADE"))
            
            print(f"🗑️  Truncating claims table...")
            await session.execute(text("TRUNCATE TABLE claims CASCADE"))
            
            await session.commit()
            
            print(f"\n✅ Knowledge base reset complete!")
            print(f"   The system will now learn fresh from the new Tavily-filtered pipeline.")
            
        except Exception as exc:
            await session.rollback()
            logger.error(f"Reset failed: {exc}")
            print(f"\n❌ Reset failed: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Reset the FactuAI knowledge base (claims + evidence tables)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Execute the reset (default is dry-run mode)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FactuAI Knowledge Base Reset")
    print("=" * 60)
    
    asyncio.run(reset_knowledge_base(dry_run=not args.force))


if __name__ == "__main__":
    main()
