"""
Analyze frontend TypeScript/TSX files for Phase 1 audit.
"""
import re
from pathlib import Path
from collections import defaultdict

def analyze_typescript_files(frontend_root):
    """Analyze all TypeScript/TSX files in frontend directory."""
    results = []
    
    for ts_file in frontend_root.rglob('*.ts*'):
        if 'node_modules' in str(ts_file) or '.next' in str(ts_file):
            continue
        
        rel_path = ts_file.relative_to(frontend_root)
        
        try:
            with open(ts_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.splitlines())
                non_empty_lines = len([l for l in content.splitlines() if l.strip()])
                
                # Analyze imports
                imports = re.findall(r'import\s+.*?from\s+["\'](.+?)["\']', content)
                
                # Count React components (export default/export const/export function)
                components = re.findall(r'export\s+(?:default\s+)?(?:function|const)\s+(\w+)', content)
                
                # Check for hooks
                hooks = re.findall(r'(?:export\s+)?(?:function|const)\s+(use\w+)', content)
                
                # Check for interfaces/types
                interfaces = re.findall(r'(?:export\s+)?interface\s+(\w+)', content)
                types = re.findall(r'(?:export\s+)?type\s+(\w+)', content)
                
                # Categorize imports
                react_imports = [i for i in imports if 'react' in i.lower()]
                next_imports = [i for i in imports if 'next' in i]
                local_imports = [i for i in imports if i.startswith('.') or i.startswith('@/')]
                
                results.append({
                    'path': str(rel_path),
                    'loc': lines,
                    'non_empty_loc': non_empty_lines,
                    'imports': len(imports),
                    'local_imports': len(local_imports),
                    'components': components,
                    'hooks': hooks,
                    'interfaces': interfaces,
                    'types': types,
                    'file_type': 'tsx' if ts_file.suffix == '.tsx' else 'ts'
                })
        except Exception as e:
            print(f"Error reading {rel_path}: {e}")
    
    # Sort by path
    results.sort(key=lambda x: x['path'])
    
    return results

def print_results(results):
    """Print analysis results."""
    print("=" * 110)
    print(f"{'PATH':<65} {'LOC':<8} {'TYPE':<6} {'COMP':<6} {'HOOKS':<7}")
    print("=" * 110)
    
    for r in results:
        path = r['path']
        if len(path) > 63:
            path = "..." + path[-60:]
        file_type = r['file_type']
        comp_count = len(r['components'])
        hook_count = len(r['hooks'])
        
        print(f"{path:<65} {r['loc']:<8} {file_type:<6} {comp_count:<6} {hook_count:<7}")
    
    print("=" * 110)
    print(f"Total files: {len(results)}")
    print(f"Total LOC: {sum(r['loc'] for r in results)}")
    print(f"Average LOC per file: {sum(r['loc'] for r in results) / len(results):.1f}")
    
    # Files under 50 LOC (merge candidates)
    small_files = [r for r in results if r['non_empty_loc'] < 50]
    print(f"\nFiles under 50 LOC (merge candidates): {len(small_files)}")
    for r in small_files[:20]:  # Show first 20
        print(f"  - {r['path']} ({r['non_empty_loc']} lines)")
    
    # Component files by directory
    print("\n\nComponent distribution by directory:")
    by_dir = defaultdict(int)
    for r in results:
        if r['file_type'] == 'tsx' and r['components']:
            dir_path = str(Path(r['path']).parent)
            by_dir[dir_path] += 1
    
    for dir_path, count in sorted(by_dir.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {dir_path}: {count} components")

if __name__ == '__main__':
    frontend_root = Path(__file__).parent.parent / 'frontend' / 'src'
    results = analyze_typescript_files(frontend_root)
    print_results(results)
    
    # Save detailed results
    output_file = Path(__file__).parent.parent / 'frontend_analysis.json'
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nDetailed results saved to: {output_file}")
