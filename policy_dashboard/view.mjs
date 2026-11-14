export const PROTOCOL='dreamer_actor_critic_imagination_v1';
const pin=x=>typeof x==='string'&&/^[0-9a-f]{64}$/.test(x);
export const numeric=x=>typeof x==='number'&&Number.isFinite(x)?x:null;
export function policyView(s={}){
  const active=s.protocol===PROTOCOL&&typeof s.run_id==='string'&&s.run_id.length>0&&pin(s.plan_sha256)
    &&pin(s.world_sha256)&&pin(s.data_identity_sha256)&&Number.isInteger(s.step)&&s.step>=0
    &&Number.isInteger(s.world_step)&&s.world_step>0;
  const matches=row=>active&&row?.protocol===PROTOCOL&&row.run_id===s.run_id&&row.plan_sha256===s.plan_sha256
    &&row.world_sha256===s.world_sha256&&row.data_identity_sha256===s.data_identity_sha256
    &&row.world_step===s.world_step
    &&Number.isInteger(row.step)&&row.step>0&&row.step<=s.step;
  const history=(Array.isArray(s.history)?s.history:[]).filter(matches).sort((a,b)=>a.step-b.step);
  const metrics=active&&s.metrics&&typeof s.metrics==='object'?s.metrics:{};
  const r=s.validation_report??s.validation??{},policy=r.policy_identity||{},world=r.world_identity||{};
  const receipt=s.validation_receipt||{},stored=s.validation_receipt_verified===true
    &&receipt.policy_checkpoint_sha256===policy.checkpoint_sha256&&receipt.policy_step===r.step
    &&receipt.plan_sha256===s.plan_sha256&&receipt.world_sha256===world.checkpoint_sha256&&receipt.world_step===world.step;
  const worldMatches=world.checkpoint_sha256===s.world_sha256&&world.step===s.world_step;
  const valid=active&&(!s.validation_receipt||stored)&&r.kind==='dreamer_policy_validation_v1'&&r.version===1&&r.training_protocol===PROTOCOL
    &&Number.isInteger(r.step)&&r.step>=0&&r.step<=s.step&&r.horizon>=1&&r.horizon<=450
    &&Number.isInteger(r.starts)&&r.starts>0&&pin(r.panel_fingerprint)
    &&r.data_identity_sha256===s.validation_data_identity_sha256&&r.panel_fingerprint===s.validation_panel_fingerprint
    &&pin(policy.checkpoint_sha256)&&policy.step===r.step
    &&policy.plan_sha256===s.plan_sha256&&(worldMatches||stored)
    &&r.learned_world_Source_feedback===false&&r.matched_action_Source_feedback===false
    &&r.Source_closedloop_validation_only===true&&r.gameplay_qualified===false
    &&r.automatic_policy_deployment===false&&['random','scripted','learned_policy'].every(name=>
      r.arms?.[name]?.cases?.length===r.starts&&r.arms[name].source_return?.n===r.starts
      &&r.arms[name].learned_predicted_head_return?.n===r.starts);
  return{active,matches,history,metrics,validation:valid?r:null,validation_valid:valid,validation_uses_previous_world:valid&&!worldMatches};
}

const names={random:'Случайные действия',scripted:'Простое правило',learned_policy:'Обученная политика'};
export function validationRows(report){
  if(!report)return[];
  return Object.entries(names).map(([name,label])=>{const a=report.arms[name];return{label,
    sourceReturn:numeric(a.source_return.mean),imaginedReturn:numeric(a.learned_predicted_head_return.mean),
    sourceKills:numeric(a.source_kills?.mean),sourceHP:numeric(a.source_final_HP?.mean),
    sourceWalls:numeric(a.source_wall_steps),learnedWalls:numeric(a.learned_wall_steps),
    drift:numeric(a.matched_final_player_drift_tiles?.mean)};});
}

export function rewardValidationView(s,policyReport){
  const r=s.reward_validation||{},m=r.metrics||{},p=policyReport?.policy_identity,w=policyReport?.world_identity;
  const valid=!!policyReport&&r.kind==='dreamer_real_Source_reward_continue_validation_v1'
    &&r.source_held_out===true&&r.split==='val'&&r.no_optimizer_updates===true&&r.RNG_preserved===true
    &&r.policy_step===policyReport.step&&r.policy_checkpoint_sha256===p?.checkpoint_sha256
    &&r.world_sha256===w?.checkpoint_sha256&&r.world_step===w?.step&&r.plan_sha256===s.plan_sha256;
  return{valid,report:r,deathRecall:valid&&m.death_transitions>0?numeric(m.death_recall):null,
    falseDeathRate:valid&&m.alive_transitions>0?numeric(m.false_death_rate):null,
    rewardMAE:valid?numeric(m.unweighted_reward_MAE??m.reward_MAE):null,
    continueBrier:valid?numeric(m.unweighted_continue_Brier??m.continue_Brier):null,
    deaths:valid?numeric(m.death_transitions):null,alive:valid?numeric(m.alive_transitions):null,
    rewardClipped:valid&&m.reward_target_target_count>0?numeric(m.reward_target_clipped_fraction):null};
}
