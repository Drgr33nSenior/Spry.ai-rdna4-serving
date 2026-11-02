"""One existing45s decoder child within unchanged75s work scope; no code grader."""
import time
from pathlib import Path
import quality_contract as q

def check_stage(row,*,argv,seconds,deadline):
 need=q.need;record=row['supervision']
 need(type(record)is dict and set(record)=={'argv','seconds','term_seconds','reap_seconds','complete','observed_processes',
  'signals','failures','pid','pgid','exit_code','child_reaped','owned_group_absent','elapsed_seconds'},'exact successful supervisor schema')
 need(type(row)is dict and set(row)=={'label','started_monotonic','finished_monotonic','supervision'}and
  type(row['started_monotonic'])in(int,float)and type(row['finished_monotonic'])in(int,float)and
  0<row['started_monotonic']<=row['finished_monotonic']<=deadline,'stage actual shared-deadline interval')
 need(record['argv']==argv and type(record['seconds'])is int and record['seconds']==seconds and
  type(record['term_seconds'])is type(record['reap_seconds'])is int and record['term_seconds']==record['reap_seconds']==1 and record['complete']is record['child_reaped']is record['owned_group_absent']is True and
  type(record['exit_code'])is int and record['exit_code']==0 and record['failures']==[]and record['signals']==[],
  'source-bound finite child completed/reaped/released without failure')
 need(type(record['pid'])is type(record['pgid'])is int and record['pid']>0 and record['pgid']==record['pid']and
  type(record['elapsed_seconds'])in(int,float)and 0<record['elapsed_seconds']<=seconds+2 and
  record['elapsed_seconds']<=row['finished_monotonic']-row['started_monotonic'],
  'bounded actual process/group and stage duration')
 observed=record['observed_processes']
 need(type(observed)is list and len(observed)==1 and set(observed[0])=={'pid','start_ticks'}and
  observed[0]['pid']==record['pid']and type(observed[0]['pid'])is type(observed[0]['start_ticks'])is int and
  observed[0]['start_ticks']>0,'actual source-owned stage process/start identity')


def run(plan,host):
 common,package=host.authenticated_common(plan);source=Path(plan['source_root']);run=Path(plan['run_root'])
 deadline=time.monotonic()+210;scope=host.own_scope(plan);proof=run/'STACK-PROOF.json'
 host.held(Path('/usr/bin/bwrap'),'7c44fa8e7326e62e81ab3f70ff682bfc0eb3b447b39cf9fbb779a31948364762',2<<20)
 admission=dict(package_sha256=plan['package_sha256'],boot_id=host.BOOT,scope=plan['unit'],**scope,
  work_deadline_monotonic=deadline,stack_proof_sha256=host.sha(host.held(proof)),
  tokenizer_runtime_inventory_sha256=host.sha(host.held(run/'TOKENIZER-RUNTIME-INVENTORY.zlib',maximum=8<<20)))
 admission_pin=host.put(run/'ADMISSION.json',admission)
 decoder_argv=common.bootstrap_bwrap(package,deadline,admission_file=run/'ADMISSION.json',admission_sha256=admission_pin,
  stack_proof_file=proof,cgroup_directory=Path('/sys/fs/cgroup'+scope['host_cgroup_path']))
 host.put(run/'INNER-ARGV.json',dict(argv=decoder_argv,deadline_monotonic=deadline,admission_sha256=admission_pin))
 stages=[];results=[];failure=None
 def child(label,argv,seconds):
  q.need(time.monotonic()+seconds+2<=deadline,'whole child and cleanup fit shared210s work deadline')
  started=time.monotonic();record=host.run_finite(argv,run,label,seconds,roots=(source/'output',),term=1,reap=1)
  row=dict(label=label,started_monotonic=started,finished_monotonic=time.monotonic(),supervision=record);stages.append(row)
  check_stage(row,argv=argv,seconds=seconds,deadline=deadline)
 try:
  child('quality-decode',decoder_argv,180)
  decoder_path=source/'output/quality/DECODE-RESULT.json'
  decoded=host.decode_result(plan,run,decoder_path)
  value=q.strict(host.held(source/'QUALITY-CPU-INPUT.json',plan['sources']['QUALITY-CPU-INPUT.json'],q.INPUT_LIMIT))
  expected=q.strict(host.held(source/'EXPECTED-QUALITY.json',plan['sources']['EXPECTED-QUALITY.json']))
  q.validate_answers(value,expected,decoded['answers'])
  grades=q.grade_answers(value,expected,decoded['answers'],results)
  final=dict(schema=1,mode=q.MODE,complete=True,qualification=False,
   input_sha256=plan['sources']['QUALITY-CPU-INPUT.json'],decode_result_sha256=host.sha(host.held(decoder_path)),
   stages=stages,python_results=results,grade_summary=grades,work_deadline_monotonic=deadline)
  host.put(source/'output/quality/QUALITY-RESULT.json',final)
 except BaseException as error:
  failure=error
  failed=dict(schema=1,mode=q.MODE,complete=False,qualification=False,input_sha256=plan['sources']['QUALITY-CPU-INPUT.json'],
   stages=stages,python_results=results,failure=type(error).__name__+': '+str(error)[:2048],work_deadline_monotonic=deadline)
  try:host.put(source/'output/QUALITY-FAILED.json',failed)
  except BaseException as secondary:error.add_note('Secondary quality-stage publication failure: '+str(secondary)[:512])
 if failure is not None:raise failure


def validate_final(result,*,plan,run,source,decoded,value,expected,host,admission):
 q.need(type(result)is dict and set(result)=={'schema','mode','complete','qualification','input_sha256',
  'decode_result_sha256','stages','python_results','grade_summary','work_deadline_monotonic'}and type(result['schema'])is int and
  result['schema']==1 and result['mode']==q.MODE and result['complete']is True and result['qualification']is False and
  result['input_sha256']==plan['sources']['QUALITY-CPU-INPUT.json']and result['work_deadline_monotonic']==admission['work_deadline_monotonic'],
  'exact final processing schema, distinct from task quality')
 q.need(result['decode_result_sha256']==host.sha(host.held(source/'output/quality/DECODE-RESULT.json',maximum=4<<20)),'actual decoder output identity')
 q.validate_answers(value,expected,decoded['answers'])
 inner=host.strict_json(host.held(run/'INNER-ARGV.json'));stages=result['stages'];code=[]
 q.need(type(stages)is list and len(stages)==1+len(code)and type(result['python_results'])is list and len(result['python_results'])==len(code),'exact decoder and available-code stages')
 prior=0;identities=[]
 for index,row in enumerate(stages):
  label='quality-decode';argv=inner['argv'];seconds=180
  q.need(row['label']==label and row['started_monotonic']>=prior,'serial original-order child chronology')
  check_stage(row,argv=argv,seconds=seconds,deadline=admission['work_deadline_monotonic']);prior=row['finished_monotonic']
  actual=host.strict_json(host.held(run/(label+'.json')));q.need(actual==row['supervision'],'retained actual child supervision join')
  identities.append(actual['pid'])
 q.need(len(set(identities))==len(identities),'distinct serial child processes')
 q.need(result['grade_summary']==q.grade_answers(value,expected,decoded['answers'],result['python_results']),'exact frozen final task grades')
 return result

