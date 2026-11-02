"""Check the actual generated final campaign sources without registration or execution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spry_common.canon import canonical_line as canonical,digest,need


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path);args=parser.parse_args()
    root=args.root.resolve(strict=True);summary=json.loads((root/'GPU-PREPARATION.json').read_bytes())
    need(summary['launch_authorized'] is False and summary['model_starts_executed']==0,'inert preparation required')
    counts={};common={}
    for variant,expected in (('baseline',78),('cleanup',79)):
        author=root/variant/'author';review=author/'package-v2/review'
        manifest=json.loads((author/'package-v2/MANIFEST.json').read_bytes())
        binding=json.loads((author/'LAUNCHER-BINDINGS-TEMPLATE.json').read_bytes())
        approval=json.loads((root/variant/'APPROVAL-REQUEST.json').read_bytes())
        need(approval['model_start_authorized'] is False and approval['maximum_starts']==0 and
             approval['cpu_execution_authorized'] is False,'unfunded request required')
        protection=json.loads((author/'package-v2/PROTECTED-SHA256.json').read_bytes())
        counts[variant]=len(manifest['package_files'])+len(list((author/'package-v2/config').rglob('*.json')))+2
        need(counts[variant]==expected,'exact final archive membership required')
        script=author/'owner-upload/RUN-ON-WORKSTATION.sh'
        result=subprocess.run(['/bin/bash','-n',str(script)],capture_output=True,timeout=10)
        need(result.returncode==0,'generated shell syntax')
        for name,pin in binding['upload_inventory'].items():
            need(digest((author/'owner-upload'/name).read_bytes())==pin,'upload pin differs: '+name)
        common[variant]='review/spry_common_canon.py' in protection
        if common[variant]:
            pin=protection['review/spry_common_canon.py'];raw=(author/'owner-upload/review/spry_common_canon.py').read_bytes()
            need(digest(raw)==binding['bootstrap_common_sha256']==pin,'standalone common pin')
            shell=script.read_text()
            need(pin+'  review/spry_common_canon.py' in shell and
                 pin+'  '+binding['protected_root']+'/review/spry_common_canon.py' in shell and
                 'benchmark_artifacts.py spry_common_canon.py; do' in shell,'complete root common staging')
        for arm in ('control','mtp'):
            source=json.loads((review/('HTTP-BINDING-'+arm+'.json')).read_bytes())['binding_template']
            cli=source['canonical_server_cli'];need(cli[cli.index('--decode-log-interval')+1]=='100','matched production logging')
    need(common=={'baseline':False,'cleanup':True},'protected common membership')
    out=dict(passed=True,source_only=True,archive_members=counts,protected_common_upload=common,
        production_decode_log_interval=100,all_upload_pins_and_shell_syntax_checked=True,
        model_starts=0,launch_authorized=False,native_execution=False)
    with args.report.open('xb') as stream:stream.write(canonical(out))
    print(json.dumps(out,sort_keys=True))


if __name__=='__main__':main()
