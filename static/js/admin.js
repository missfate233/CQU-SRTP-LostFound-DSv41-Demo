/* ==========================================================================
   管理页：删除记录、全量重算匹配
   ========================================================================== */
'use strict';

(function () {
  // ---- 删除记录 ----
  document.querySelectorAll('[data-action="delete-item"]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const itemId = btn.dataset.itemId;
      const row = btn.closest('tr');
      const title = row ? (row.querySelector('a') || {}).textContent || '' : '';
      if (!window.confirm(`确认删除这条记录吗？\n\n${title}\n\n记录及其图片将被永久删除。`)) return;
      btn.disabled = true;
      btn.textContent = '删除中…';
      try {
        await LF.api(`/api/items/${itemId}`, { method: 'DELETE' });
        if (row) {
          row.style.transition = 'opacity .25s';
          row.style.opacity = '0';
          setTimeout(() => row.remove(), 260);
        }
        LF.toast('已删除', 'ok');
      } catch (err) {
        LF.toast(err.message, 'error');
        btn.disabled = false;
        btn.textContent = '删除';
      }
    });
  });

  // ---- 全量重算 ----
  const recomputeBtn = document.getElementById('recompute-btn');
  if (recomputeBtn) {
    recomputeBtn.addEventListener('click', async () => {
      if (!window.confirm('将清空并重新计算所有「失物 × 拾物」组合的匹配得分，确定继续？')) return;
      recomputeBtn.disabled = true;
      recomputeBtn.textContent = '计算中…';
      try {
        const data = await LF.api('/api/match/recompute', { method: 'POST', body: { clear: true } });
        const r = data.result;
        LF.toast(
          `重算完成：比较 ${r.combinations} 组，命中 ${r.matches} 组（高可信 ${r.strong} 组），耗时 ${r.seconds}s`,
          'ok',
          5000
        );
        setTimeout(() => window.location.reload(), 1400);
      } catch (err) {
        LF.toast(err.message, 'error');
        recomputeBtn.disabled = false;
        recomputeBtn.textContent = '重算全库匹配';
      }
    });
  }
})();
