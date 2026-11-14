import test from'node:test';
import assert from'node:assert/strict';
import{policyView,validationRows,rewardValidationView,numeric,PROTOCOL}from'./view.mjs';
import{readFile}from'node:fs/promises';
import vm from'node:vm';

function fixture(){
  const pin=n=>n.repeat(64),s={protocol:PROTOCOL,run_id:'separate_policy',plan_sha256:pin('a'),world_sha256:pin('b'),
    data_identity_sha256:pin('c'),validation_data_identity_sha256:pin('d'),validation_panel_fingerprint:pin('e'),
    step:20,world_step:6746,checkpoint_step:20,checkpoint_sha256:pin('9'),metrics:{actor_loss:0,critic_loss:1.2,imagined_return:99}};
  s.history=[{protocol:PROTOCOL,run_id:s.run_id,plan_sha256:s.plan_sha256,world_sha256:s.world_sha256,
    data_identity_sha256:s.data_identity_sha256,world_step:s.world_step,step:10,metrics:{actor_loss:-.1,critic_loss:2}}];
  s.validation={kind:'dreamer_policy_validation_v1',version:1,training_protocol:PROTOCOL,step:10,horizon:32,starts:4,
    panel_fingerprint:s.validation_panel_fingerprint,data_identity_sha256:s.validation_data_identity_sha256,
    policy_identity:{checkpoint_sha256:pin('f'),step:10,plan_sha256:s.plan_sha256},
    world_identity:{checkpoint_sha256:s.world_sha256,step:s.world_step},
    learned_world_Source_feedback:false,matched_action_Source_feedback:false,Source_closedloop_validation_only:true,
    gameplay_qualified:false,automatic_policy_deployment:false,
    arms:Object.fromEntries(['random','scripted','learned_policy'].map(name=>[name,{cases:[{},{},{},{}],source_return:{n:4,mean:1},
      learned_predicted_head_return:{n:4,mean:99},source_kills:{mean:1},source_final_HP:{mean:80},source_wall_steps:0,
      learned_wall_steps:3,matched_final_player_drift_tiles:{mean:.5}}]))};return s;
}

test('policy page separates actual Source reward from predicted imagination and preserves real zero loss',()=>{
  const s=fixture(),v=policyView(s);assert.equal(v.active,true);assert.equal(v.validation_valid,true);
  assert.equal(v.metrics.actor_loss,0);assert.equal(v.history.length,1);
  const rows=validationRows(v.validation);assert.equal(rows[2].sourceReturn,1);assert.equal(rows[2].imaginedReturn,99);
  assert.equal(rows[2].sourceWalls,0);assert.equal(rows[2].learnedWalls,3);
  assert.equal(v.validation.gameplay_qualified,false);assert.equal(v.validation.automatic_policy_deployment,false);
});

test('policy history uses only current run plan world and TRAIN data while VAL has a separate data pin',()=>{
  for(const mutate of[r=>r.protocol='old',r=>r.run_id='other',r=>r.plan_sha256='old',r=>r.world_sha256='other',
    r=>r.data_identity_sha256='old',r=>r.world_step++,r=>r.step=21,r=>r.step=0]){
    const s=fixture();mutate(s.history[0]);assert.equal(policyView(s).history.length,0);
  }
  const s=fixture();assert.notEqual(s.data_identity_sha256,s.validation_data_identity_sha256);
  assert.equal(policyView(s).validation_valid,true);s.validation.data_identity_sha256=s.data_identity_sha256;
  assert.equal(policyView(s).validation_valid,false);
});

test('last completed validation keeps its own saved policy checkpoint when current learning advances',()=>{
  const s=fixture();assert.notEqual(s.checkpoint_sha256,s.validation.policy_identity.checkpoint_sha256);
  assert.equal(policyView(s).validation_valid,true);
  const r=s.validation;s.validation_receipt={policy_checkpoint_sha256:r.policy_identity.checkpoint_sha256,
    policy_step:r.step,plan_sha256:s.plan_sha256,world_sha256:r.world_identity.checkpoint_sha256,world_step:r.world_identity.step};
  s.validation_receipt_verified=true;s.world_sha256='1'.repeat(64);s.world_step=6750;
  const v=policyView(s);assert.equal(v.validation_valid,true);assert.equal(v.validation_uses_previous_world,true);
  s.validation_receipt_verified=false;assert.equal(policyView(s).validation_valid,false);
});

test('unrelated or corrected world evaluations never count as verified real policy performance',()=>{
  for(const mutate of[r=>r.step=21,r=>r.world_identity.checkpoint_sha256='other',r=>r.panel_fingerprint='other',
    r=>r.policy_identity.plan_sha256='other',r=>r.learned_world_Source_feedback=true,
    r=>r.matched_action_Source_feedback=true,r=>r.gameplay_qualified=true,r=>r.arms.random.cases.pop()]){
    const s=fixture();mutate(s.validation);assert.equal(policyView(s).validation_valid,false);
  }
  const s=fixture();s.protocol='rssm_main18_full_bptt_loss_ladder_v1';assert.equal(policyView(s).active,false);
});

test('reward calibration reports genuine support zero recall and overlapping windows without invented success',()=>{
  const s=fixture(),p=s.validation;s.reward_validation={kind:'dreamer_real_Source_reward_continue_validation_v1',
    source_held_out:true,split:'val',no_optimizer_updates:true,RNG_preserved:true,policy_step:p.step,
    policy_checkpoint_sha256:p.policy_identity.checkpoint_sha256,world_sha256:p.world_identity.checkpoint_sha256,
    world_step:p.world_identity.step,plan_sha256:s.plan_sha256,metrics:{death_transitions:18,alive_transitions:558,
      death_recall:0,false_death_rate:.02,unweighted_reward_MAE:.1,unweighted_continue_Brier:.05,
      reward_target_target_count:576,reward_target_clipped_fraction:0}};
  let v=rewardValidationView(s,p);assert.equal(v.valid,true);assert.equal(v.deathRecall,0);assert.equal(v.rewardClipped,0);
  s.reward_validation.metrics.death_transitions=0;v=rewardValidationView(s,p);assert.equal(v.deathRecall,null);
  s.reward_validation.policy_step++;assert.equal(rewardValidationView(s,p).valid,false);
});

test('browser module renders pinned progress and actual Source table with no game control action',async()=>{
  const html=await readFile(new URL('./index.html',import.meta.url),'utf8'),nodes=new Map(),ids=new Set([...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]));
  const element=()=>({children:[],textContent:'',style:{},append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x},setAttribute(){},addEventListener(){}});
  const s=fixture(),context=vm.createContext({policyView,validationRows,rewardValidationView,numeric,console,
    fetch:async()=>({ok:true,json:async()=>s}),setInterval:()=>0,
    document:{getElementById(id){assert.ok(ids.has(id));if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)},
      createElement:element,createElementNS:element}});
  const js=html.match(/<script type="module">([\s\S]*?)<\/script>/)[1].replace(/^import[^\n]+\n/m,'');
  vm.runInContext(js,context);await vm.runInContext('refresh()',context);
  assert.match(nodes.get('status').textContent,/20/);assert.equal(nodes.get('metrics').children[0].children[1].textContent,'0');
  assert.equal(nodes.get('validationRows').children.length,3);
  assert.equal(nodes.get('validationRows').children[2].children[1].textContent,'1');
  assert.equal(nodes.get('validationRows').children[2].children[2].textContent,'99');
  assert.match(nodes.get('validationStatus').textContent,/32.*не заявлено/);
  const r=s.validation;s.validation_receipt={policy_checkpoint_sha256:r.policy_identity.checkpoint_sha256,
    policy_step:r.step,plan_sha256:s.plan_sha256,world_sha256:r.world_identity.checkpoint_sha256,world_step:r.world_identity.step};
  s.validation_receipt_verified=true;s.world_step=6750;s.world_sha256='1'.repeat(64);
  await vm.runInContext('refresh()',context);
  assert.match(nodes.get('validationStatus').textContent,/Оценён world 6746; текущий world 6750.*прежней world model/);
  assert.match(html,/Imagined return не заменяет|награду Source/);
  assert.doesNotMatch(html,/api\/.*(?:play|control|launch)|method:\s*['"]POST/);
});
