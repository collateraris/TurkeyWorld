"""Held-out visual-first detection metrics, including absent and wrong hints."""
import torch
from .enemy_decoder_sr import box_iou


def wrong_prediction_hint(hypothesis):
    """Corrupt only cached predictions; never derive a hint from Source labels."""
    if hypothesis is None:
        return None
    wrong = hypothesis.detach().roll(1, dims=0).clone()
    if len(wrong) == 1:
        wrong[:, 8:10] = (2*wrong[:, :2]-wrong[:, 8:10]).clamp(0, 1)
    wrong[:, -1] = hypothesis[:, -1]
    return wrong


class VisualFirstDiagnostics:
    def __init__(self):
        self.domains = {}
        self.invariance = dict(frames=0, no_hint_max_RGB_error=0., wrong_hint_max_RGB_error=0.)

    def add(self, mode, output, batch, image_rows):
        data = self.domains.setdefault(mode, dict(rows=[],tp=0,fp=0,fn=0,tn=0,
            visual_route_frames=0,guided_route_frames=0,bbox_iou_sum=0.,bbox_iou_frames=0))
        visible = batch['mask_hr'].detach().bool().flatten(1).any(-1)
        visual = output['visual_route'].detach().reshape(-1) > .5
        guided = output['guided_route'].detach().reshape(-1) > .5
        detected = visual | guided
        for key, selected in (('tp', detected & visible), ('fp', detected & ~visible),
                              ('fn', ~detected & visible), ('tn', ~detected & ~visible)):
            data[key] += int(selected.sum())
        data['visual_route_frames'] += int(visual.sum())
        data['guided_route_frames'] += int(guided.sum())
        boxes = torch.where(visual[:, None], output['visual_bbox'], output['enemy_bbox'])
        iou = box_iou(boxes.detach().float(), batch['bbox_hr'].detach().float())
        data['bbox_iou_sum'] += float((iou*detected)[visible].sum())
        data['bbox_iou_frames'] += int(visible.sum())
        data['rows'].extend(image_rows)

    def compare(self, main, alternatives):
        selected = main['visual_route'].detach().reshape(-1) > .5
        self.invariance['frames'] += int(selected.sum())
        if not bool(selected.any()):
            return
        for mode, output in alternatives.items():
            error = float((main['image'][selected]-output['image'][selected]).abs().max())
            key = mode+'_max_RGB_error'
            self.invariance[key] = max(self.invariance[key],error)

    def report(self):
        from .semantic_sr_validation import _summary
        result = {}
        for mode, data in self.domains.items():
            summary = _summary(data['rows'], [True]*len(data['rows']))
            positive,negative = data['tp']+data['fn'],data['fp']+data['tn']
            result[mode] = dict(frames=len(data['rows']),positive_frames=positive,negative_frames=negative,
                **{key:data[key]for key in ('tp','fp','fn','tn','visual_route_frames','guided_route_frames')},
                confirmation_recall=data['tp']/max(1,positive),
                confirmation_false_positive_rate=data['fp']/max(1,negative),
                bbox_iou_mean=data['bbox_iou_sum']/max(1,data['bbox_iou_frames']),
                bbox_iou_frames=data['bbox_iou_frames'],
                **{key:summary.get(key) for key in ('mae_mean','ssim_mean','foreground_mae','roi_crop_ssim_mean')})
        invariant = dict(self.invariance)
        invariant['passed'] = (all(invariant[key] == 0 for key in ('no_hint_max_RGB_error','wrong_hint_max_RGB_error'))
                               if invariant['frames'] else None)
        return dict(domains=result, scope='all_held_out_NPR_HR_visible_empty',
            wrong_hint_policy='deterministic_roll_of_cached_predicted_states_with_original_availability',
            high_visual_output_invariance=invariant, Source_future_state_as_input=False)
