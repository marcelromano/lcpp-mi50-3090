"""CPU HTTP contract tests. Launch only inside the enclosing gpu-run ticket.
No production engine/service is stopped; the binary is CPU-only and bound to loopback.
"""
import json, os, pathlib, socket, struct, subprocess, tempfile, time, unittest, urllib.request, urllib.error, uuid
ROOT=pathlib.Path(os.environ.get('T1977_ROOT','/tmp'))
BIN=pathlib.Path(os.environ.get('LLAMA_SERVER_BIN_PATH',str(ROOT/'art/llama-server')))
MODEL=pathlib.Path(os.environ.get('LLAMA_B2_TEST_MODEL',str(ROOT/'fixtures/attn.gguf')))
@unittest.skipUnless(os.environ.get('LLAMA_B2_TEST_MODEL'), 'set LLAMA_B2_TEST_MODEL to the fixed-seed CPU fixture')
class Contract(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='b2-',dir=ROOT/'evidence')
  self.log=open(pathlib.Path(self.tmp.name)/'server.log','w')
  with socket.socket() as s: s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
  self.url=f'http://127.0.0.1:{port}'
  self.args=[str(BIN),'-m',str(MODEL),'-ngl','0','-t','2','-tb','2','-c','512','-b','64','-ub','64','-np','2','--kv-unified','--cache-ram','0','--no-cache-idle-slots','--host','127.0.0.1','--port',str(port),'--slots','--metrics','--slot-save-path',self.tmp.name,'--api-key','fixture-not-a-secret']
  self.proc=None
  self.env=os.environ.copy()
 def tearDown(self):
  if self.proc and self.proc.poll() is None:
   self.proc.terminate()
   try:self.proc.wait(timeout=10)
   except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
  self.log.close()
  # Retain logs as evidence, remove generated snapshot data.
  dest=ROOT/'evidence'/f'{self.id().split(".")[-1]}.server.log'
  dest.write_bytes((pathlib.Path(self.tmp.name)/'server.log').read_bytes())
  self.tmp.cleanup()
 def request(self,path,body=None,auth=True):
  headers={'Content-Type':'application/json'}
  if auth:headers['Authorization']='Bearer fixture-not-a-secret'
  req=urllib.request.Request(self.url+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
  try:r=urllib.request.urlopen(req,timeout=30)
  except urllib.error.HTTPError as e:r=e
  data=r.read();return r.status,json.loads(data) if data and r.headers.get('Content-Type','').startswith('application/json') else data
 def start(self,*extra):
  self.proc=subprocess.Popen(self.args+list(extra),stdout=self.log,stderr=subprocess.STDOUT,env=self.env)
  with open(ROOT/'evidence/testserver-pids.txt','a') as pids:pids.write(str(self.proc.pid)+'\n')
  # Bounded readiness wait is inside the ticket/test process, not agent polling.
  deadline=time.monotonic()+20
  while time.monotonic()<deadline:
   if self.proc.poll() is not None:raise AssertionError(f'server exited {self.proc.returncode}')
   try:
    if self.request('/health')[0]==200:return
   except OSError:pass
   time.sleep(.05)
  raise AssertionError('readiness deadline')
 def completion(self,slot=0,prompt='the quick brown fox',n=8):
  request={'prompt':prompt,'id_slot':slot,'n_predict':n,'temperature':0,'seed':42,'cache_prompt':True}
  if hasattr(self,'completion_bias'):request['logit_bias']=self.completion_bias
  code,body=self.request('/completion',request)
  self.assertEqual(code,200,body);return body
 def operation(self,action,seq,slot=0,filename='state.bin',ack=True,**override):
  code,cap=self.request('/slot-operations');self.assertEqual(code,200,cap)
  body={k:cap[k] for k in ['engine_id','owner','engine_epoch']}
  body.update(op_id=str(uuid.uuid4()),owner_seq=seq)
  if action!='erase':body['filename']=filename
  body.update(override)
  code,result=self.request(f'/slots/{slot}?action={action}',body)
  if override:return code,result
  self.assertEqual(code,200,result);self.assertEqual(result['status'],'completed',result)
  # exact retries must not reexecute even after the slot changes
  self.assertEqual(self.request(f'/slots/{slot}?action={action}',body),(200,result))
  self.assertEqual(self.request('/slot-operations/'+body['op_id']),(200,result))
  if ack:
   self.assertEqual(self.request('/slot-operations/'+body['op_id']+'/ack',body)[0],204)
   self.assertEqual(self.request('/slot-operations/'+body['op_id']+'/ack',body)[0],204)
  return body,result
 def test_all_scope_admission(self):
  self.start('--slot-operation-engine-id','fixture-engine','--kv-admission','--kv-admission-restore')
  props=self.request('/props')[1]
  self.assertTrue(props['kv_admission']['enabled'])
  self.assertTrue(props['kv_admission']['supported'])
  self.assertEqual(props['kv_admission']['admission_scope'],'all')
  self.completion(0);self.completion(1,'another prefix')
  code,body=self.request('/completion',{'prompt':'the quick brown fox','id_slot':0,'n_predict':4096,'cache_prompt':True})
  self.assertEqual(code,503,body)
  self.assertEqual(body['error']['type'],'kv_capacity_error')
  # A refusal must not leave a live reservation that fences subsequent normal work.
  self.completion(0)
 def test_native_v4_same_version(self):
  self.start()
  first=self.completion()
  code,saved=self.request('/slots/0?action=save',{'filename':'native.bin'})
  self.assertEqual(code,200,saved)
  path=pathlib.Path(self.tmp.name)/'native.bin'
  self.assertEqual(struct.unpack_from('<I',path.read_bytes(),4)[0],4)
  self.assertEqual(saved['n_written'],path.stat().st_size)
  self.assertEqual(self.request('/slots/0?action=erase',{})[0],200)
  code,restored=self.request('/slots/1?action=restore',{'filename':'native.bin'})
  self.assertEqual(code,200,restored);self.assertEqual(saved['n_saved'],restored['n_restored'])
  self.assertEqual(restored['n_read'],path.stat().st_size)
  tail='the quick brown fox'+first['content']+' and'
  a=self.completion(1,tail);b=self.completion(0,tail)
  self.assertEqual(a['content'],b['content'])
  self.assertLess(a['timings']['prompt_n'],b['timings']['prompt_n'])
 def test_protocol_fences_and_api(self):
  self.start('--slot-operation-engine-id','fixture-engine')
  self.completion()
  code,cap=self.request('/slot-operations');self.assertEqual(code,200,cap)
  self.assertEqual(cap['capability'],'op_id-terminal-v1')
  self.assertEqual(cap['state_contract']['state_version'],4)
  self.assertFalse(cap['state_contract']['compat_v2'])
  self.assertTrue(cap['state_contract']['contract'].startswith('mainline-v1.0.0+fork-'))
  self.assertNotIn('unknown',cap['state_contract']['contract'])
  self.assertEqual(self.request('/slot-operations',auth=False)[0],401)
  self.assertEqual(self.request('/props')[1]['state_contract'],cap['state_contract'])
  body,saved=self.operation('save',1,ack=False)
  conflict=dict(body,filename='different.bin')
  self.assertEqual(self.request('/slots/0?action=save',conflict)[0],409)
  self.assertEqual(self.request('/slots/0?action=erase',dict(body,op_id=str(uuid.uuid4()),owner_seq=2))[0],409)
  self.assertEqual(self.request('/slot-operations/'+body['op_id']+'/ack',body)[0],204)
  self.operation('erase',2)
  _,restored=self.operation('restore',3,slot=1)
  self.assertEqual(saved['result']['n_saved'],restored['result']['n_restored'])
  self.assertEqual(self.request('/slots/0?action=save',{'filename':'untracked.bin'})[0],400)
  self.assertEqual(self.operation('erase',4,engine_epoch=str(uuid.uuid4()))[0],409)
 def test_admitted_restore_v4_and_reject_v2(self):
  self.start('--slot-operation-engine-id','fixture-engine','--kv-admission','--kv-admission-scope','restore','--kv-admission-restore')
  self.completion();self.operation('save',1)
  self.operation('erase',2);self.operation('restore',3)
  self.completion(1,'a resident prefix must survive rejected restore')
  before=next(s for s in self.request('/slots')[1] if s['id']==1)['kv_capacity']['seq_cells']
  path=pathlib.Path(self.tmp.name)/'state.bin'
  data=bytearray(path.read_bytes());struct.pack_into('<I',data,4,2);path.write_bytes(data)
  code,cap=self.request('/slot-operations');self.assertEqual(code,200)
  body={k:cap[k] for k in ['engine_id','owner','engine_epoch']}
  body.update(op_id=str(uuid.uuid4()),owner_seq=4,filename='state.bin')
  code,result=self.request('/slots/1?action=restore',body)
  self.assertEqual(code,200,result);self.assertEqual(result['status'],'failed',result)
  # Rejection is terminal and fenced; native v4 bytes are never mislabeled as v2.
  self.assertEqual(self.request('/slot-operations/'+body['op_id'])[1]['status'],'failed')
  after=next(s for s in self.request('/slots')[1] if s['id']==1)['kv_capacity']['seq_cells']
  self.assertEqual(before,after)
 def test_hybrid_device_checkpoint_disk_roundtrip(self):
  hybrid=MODEL.with_name('hybrid.gguf')
  if not hybrid.exists():self.skipTest('hybrid CPU fixture unavailable')
  self.args[self.args.index('-m')+1]=str(hybrid)
  self.env['LLAMA_CTX_CHECKPOINTS_DEVICE']='1'
  # HTTP checks file/lease/cache mechanisms; use valid ASCII output from the random
  # byte-fallback fixture. Numerical discrimination is tested separately in C++.
  self.completion_bias=[[260,100.0]]
  self.start('--slot-operation-engine-id','fixture-engine','--ctx-checkpoints','4','--checkpoint-min-step','8','-lv','5')
  prompt='the quick brown fox '*5
  first=self.completion(0,prompt)
  second=self.completion(1,prompt)
  self.assertEqual(first['content'],second['content'])
  self.operation('save',1)
  self.operation('erase',2)
  self.operation('restore',3)
  tail=prompt+first['content']+' and'
  a=self.completion(0,tail);b=self.completion(1,tail)
  self.assertEqual(a['content'],b['content'])
  self.assertEqual(a['timings']['prompt_n'],b['timings']['prompt_n'])
  self.log.flush()
  self.assertIn('storage=device',(pathlib.Path(self.tmp.name)/'server.log').read_text(errors='replace'))
 def test_epoch_restart_rejects_old_writer(self):
  extra=('--slot-operation-engine-id','fixture-engine')
  self.start(*extra);self.completion()
  body,_=self.operation('save',1,ack=False)
  self.proc.terminate();self.proc.wait(timeout=10)
  self.start(*extra)
  cap=self.request('/slot-operations')[1]
  self.assertNotEqual(cap['engine_epoch'],body['engine_epoch'])
  self.assertEqual(self.request('/slots/0?action=save',body)[0],409)
  self.assertEqual(self.request('/slot-operations/'+body['op_id']+'/ack',body)[0],204)
  self.operation('erase',1)
if __name__=='__main__':unittest.main(verbosity=2)
