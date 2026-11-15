"""Background calculations return candidates; only the GUI commits edits."""
from .anatomy import identify_bones
from .preprocessing import patient_region
from .segmentation import initial_segmentation
from .recovery import recover_ct_edges


def prepare_initial(session, parameters, progress=lambda *_: None):
    if session.joint=='leg':
        from .long_scan import prepare_whole_leg
        return prepare_whole_leg(session,parameters,progress)
    keep, report = None, None
    if parameters.get('remove_table', True):
        keep, report = patient_region(session.volume, session.roi,
                                       lambda value, text: progress(int(value * .35), text))
    result = initial_segmentation(session.volume, session.roi, parameters['threshold'],
                                  parameters['upper'], parameters['min_mm3'], parameters['fill'],
                                  lambda value, text: progress(35 + int(value * .4), text), patient_mask=keep, cleanup_report=report)
    repair = None
    if parameters.get('recover_edges', True):
        from dataclasses import replace
        from .model import Segment, PALETTE
        temporary = replace(session, parameters=dict(parameters),
            segments=[Segment(i,f'骨块 {i:02d}',PALETTE[(i-1)%len(PALETTE)]) for i in range(1,result[1]+1)])
        completed,repair = recover_ct_edges(temporary,range(1,result[1]+1),
            lambda value,text:progress(75+int(value*.2),text),labels=result[0],patient_mask=keep,
            respect_reviewed_gaps=False)
        result = (completed,result[1])
    proposals = []
    if parameters.get('auto_name', True):
        proposals = identify_bones(session.volume, result[0], session.joint, session.side,
                                   tuple(s.start for s in session.roi),
                                   lambda value, text: progress(95 + int(value * .05), text))
    return {'segmentation': result, 'recognition': proposals, 'preprocessing': report, 'repair': repair}


def recognize_existing(session, progress=lambda *_: None):
    if not session.segments:
        raise ValueError('请先运行自动初分割。')
    if session.joint=='leg':
        raise ValueError('全腿模式保留连通骨块，请对照关节和足部切片后手工命名。')
    return identify_bones(session.volume, session.labels[session.roi], session.joint, session.side,
                          tuple(s.start for s in session.roi), progress)


def install_recognition(session, proposals):
    updates = []
    summary = {'identified': 0, 'connection': 0, 'uncertain': 0, 'manual_preserved': 0}
    for proposal in proposals:
        segment = session.segment(proposal['id'])
        if segment is None:
            continue
        if segment.reviewed:
            summary['manual_preserved'] += 1
            continue
        key = 'connection' if proposal['status']=='check_connection' else ('uncertain' if proposal['status']=='uncertain' else 'identified')
        summary[key] += 1
        updates.append({'id': segment.id, 'name': proposal['name'] or f'骨块 {segment.id:02d} · 待识别',
                        'recognition': proposal})
    summary['changed'] = session.update_bone_names(updates)
    return summary
