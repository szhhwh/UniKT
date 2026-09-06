<template>
  <div class="resource-monitor">
    <div class="page-header">
      <div class="page-title-group">
        <h2 class="page-title">{{ t('route.title.resources') }}</h2>
        <span class="page-sub">{{ t('resources.pageSub') }}</span>
      </div>
      <div class="live-badge" :class="{ offline: !live }">
        <span class="live-dot"></span>
        <span class="live-text">{{ live ? t('resources.liveBadge') : t('resources.offline') }}</span>
      </div>
    </div>

    <el-skeleton :loading="loading" animated>
      <template #template>
        <div class="sys-grid">
          <div class="res-card" v-for="i in 4" :key="i">
            <el-skeleton-item variant="text" style="width:40%;height:16px" />
            <el-skeleton-item variant="text" style="width:100%;height:110px;margin-top:12px" />
          </div>
        </div>
      </template>
      <template #default>
        <div class="sys-grid">
          <div class="res-card">
            <div class="res-card-header">
              <span class="res-title">{{ t('resources.cpu') }}</span>
              <span class="res-value" v-if="snapshot">{{ snapshot.cpu_percent.toFixed(0) }}%</span>
            </div>
            <div class="core-grid" v-if="snapshot && snapshot.cpu_cores.length">
              <div
                v-for="(c, i) in coreBars"
                :key="i"
                class="core-bar"
                :title="`${c.label}: ${c.value.toFixed(0)}%`"
              >
                <div
                  class="core-fill"
                  :style="{ height: Math.max(c.value, 3) + '%', background: progressColor(c.value) }"
                ></div>
              </div>
            </div>
            <div class="chart-wrap">
              <VChart
                v-if="trendReady"
                :ref="setChart('cpu')"
                class="trend-chart"
                :option="cpuOption"
                :autoresize="true"
                @zr:dblclick="resetZoom('cpu')"
              />
              <div v-else class="trend-placeholder">{{ t('resources.collecting') }}</div>
            </div>
            <div class="res-footer" v-if="snapshot">
              <span class="res-meta">{{ t('resources.load') }}</span>
              <span class="res-meta mono">{{ snapshot.load_1m.toFixed(2) }} / {{ snapshot.load_5m.toFixed(2) }} / {{ snapshot.load_15m.toFixed(2) }}</span>
            </div>
          </div>

          <div class="res-card">
            <div class="res-card-header">
              <span class="res-title">{{ t('resources.mem') }}</span>
              <span class="res-value" v-if="snapshot">
                {{ snapshot.memory_used_gb.toFixed(1) }}/{{ snapshot.memory_total_gb.toFixed(0) }} GB
              </span>
            </div>
            <div class="chart-wrap">
              <VChart
                v-if="trendReady"
                :ref="setChart('mem')"
                class="trend-chart"
                :option="memOption"
                :autoresize="true"
                @zr:dblclick="resetZoom('mem')"
              />
              <div v-else class="trend-placeholder">{{ t('resources.collecting') }}</div>
            </div>
            <div class="res-footer" v-if="snapshot">
              <span class="res-meta">{{ t('resources.swap') }}</span>
              <span class="res-meta mono">
                {{ snapshot.swap_used_gb.toFixed(1) }}/{{ snapshot.swap_total_gb.toFixed(0) }} GB · {{ snapshot.swap_percent.toFixed(0) }}%
              </span>
            </div>
          </div>

          <div class="res-card">
            <div class="res-card-header">
              <span class="res-title">{{ t('resources.net') }}</span>
              <span class="res-value mono" v-if="trendReady">
                ↓{{ fmtRate(last(sys.netDown)) }} · ↑{{ fmtRate(last(sys.netUp)) }}
              </span>
            </div>
            <div class="chart-wrap">
              <VChart
                v-if="trendReady"
                :ref="setChart('net')"
                class="trend-chart"
                :option="netOption"
                :autoresize="true"
                @zr:dblclick="resetZoom('net')"
              />
              <div v-else class="trend-placeholder">{{ t('resources.collecting') }}</div>
            </div>
          </div>

          <div class="res-card">
            <div class="res-card-header">
              <span class="res-title">{{ t('resources.disk') }}</span>
              <span class="res-value mono" v-if="trendReady">
                R {{ fmtRate(last(sys.diskRead)) }} · W {{ fmtRate(last(sys.diskWrite)) }}
              </span>
            </div>
            <div class="chart-wrap">
              <VChart
                v-if="trendReady"
                :ref="setChart('disk')"
                class="trend-chart"
                :option="diskOption"
                :autoresize="true"
                @zr:dblclick="resetZoom('disk')"
              />
              <div v-else class="trend-placeholder">{{ t('resources.collecting') }}</div>
            </div>
          </div>
        </div>

        <div class="section-label">{{ t('resources.gpuSection') }}</div>

        <div class="gpu-grid" v-if="status && status.gpus.length > 0">
          <div class="gpu-card" v-for="gpu in status.gpus" :key="gpu.index">
            <div class="gpu-card-header">
              <span class="gpu-index">GPU {{ gpu.index }}</span>
              <span class="gpu-name" :title="gpu.name">{{ gpu.name }}</span>
            </div>

            <div class="stats-grid">
              <div class="stat-block">
                <div class="stat-label">{{ t('gpu.utilization') }}</div>
                <div class="progress-track">
                  <div
                    class="progress-fill"
                    :style="{
                      width: gpu.utilization_percent + '%',
                      background: progressColor(gpu.utilization_percent),
                    }"
                  ></div>
                </div>
                <div class="stat-value">{{ gpu.utilization_percent.toFixed(0) }}%</div>
              </div>

              <div class="stat-block">
                <div class="stat-label">{{ t('gpu.memory') }}</div>
                <div class="progress-track">
                  <div
                    class="progress-fill"
                    :style="{
                      width: (gpu.memory_total_mb > 0 ? (gpu.memory_used_mb / gpu.memory_total_mb * 100) : 0) + '%',
                      background: progressColor(gpu.memory_used_mb / (gpu.memory_total_mb || 1) * 100),
                    }"
                  ></div>
                </div>
                <div class="stat-value">
                  {{ gpu.memory_used_mb.toFixed(0) }} /
                  {{ gpu.memory_total_mb.toFixed(0) }} MB
                </div>
              </div>

              <div class="stat-block">
                <div class="stat-label">{{ t('gpu.temperature') }}</div>
                <div class="stat-row">
                  <span
                    class="temp-indicator"
                    :style="{ background: tempColor(gpu.temperature_c) }"
                  ></span>
                  <span class="stat-value">{{ gpu.temperature_c }}°C</span>
                </div>
              </div>

              <div class="stat-block">
                <div class="stat-label">{{ t('gpu.power') }}</div>
                <div class="stat-value">{{ gpu.power_usage_w.toFixed(1) }}W</div>
              </div>
            </div>

            <div class="gpu-trend">
              <VChart
                v-if="trendReady"
                :ref="setChart(`gpu-${gpu.index}`)"
                class="trend-chart"
                :option="gpuOption(gpu.index)"
                :autoresize="true"
                @zr:dblclick="resetZoom(`gpu-${gpu.index}`)"
              />
            </div>

            <div class="occupancy">
              <div class="occupancy-label">{{ t('gpu.occupancyTasks') }}</div>
              <div v-if="gpu.processes.length === 0" class="occupancy-empty">{{ t('gpu.idle') }}</div>
              <div v-else class="occupancy-list">
                <div v-for="p in gpu.processes" :key="p.id" class="occ-item">
                  <span class="occ-dot" :class="`occ-${p.status}`"></span>
                  <span class="occ-name" :title="p.name">{{ p.name }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="empty-state" v-else-if="status">
          <el-icon :size="40" class="empty-icon"><Monitor /></el-icon>
          <div class="empty-text">{{ t('gpu.notDetected') }}</div>
          <div class="empty-sub">{{ t('gpu.emptyDriverHint') }}</div>
        </div>
      </template>
    </el-skeleton>

    <div class="updated-at" v-if="status">
      {{ t('resources.updatedAt', { time: formatDateTime(status.updated_at) }) }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref, shallowRef, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDark } from '@vueuse/core'
import { useQuery } from '@tanstack/vue-query'
import { Monitor } from '@element-plus/icons-vue'
import { VChart } from '@/plugins/echarts'
import { getGpuStatus } from '@/api/gpu'
import { getResourceHistory, type ResourceSnapshot } from '@/api/resource'
import { formatDateTime } from '@/utils/date'

const { t } = useI18n()

const MAX_POINTS = 460
const lastTs = ref<number | undefined>(undefined)

const sys = reactive({
  ts: [] as number[],
  cpu: [] as number[],
  mem: [] as number[],
  swap: [] as number[],
  netUp: [] as number[],
  netDown: [] as number[],
  diskRead: [] as number[],
  diskWrite: [] as number[],
})
const gpuHist = reactive<Record<number, { util: (number | null)[]; vram: (number | null)[] }>>({})
const snapshot = shallowRef<ResourceSnapshot | null>(null)

const { data: history, isPending: historyPending, isError, dataUpdatedAt } = useQuery({
  queryKey: ['resource-history'],
  queryFn: () => getResourceHistory(lastTs.value),
  refetchInterval: 2000,
})

// A poll failing OR no fresh data for ~4 intervals means the feed is down.
const STALE_MS = 8000
const now = ref(Date.now())
let nowTimer: ReturnType<typeof setInterval> | null = null
onMounted(() => {
  nowTimer = setInterval(() => {
    now.value = Date.now()
  }, 1000)
})
onUnmounted(() => {
  if (nowTimer) clearInterval(nowTimer)
})
const live = computed(
  () =>
    !isError.value &&
    (dataUpdatedAt.value === 0 || now.value - dataUpdatedAt.value < STALE_MS),
)

watch(history, (d) => {
  if (!d) return
  const cursor = lastTs.value ?? -1
  const fresh: number[] = []
  for (let i = 0; i < d.timestamps.length; i++) {
    if (d.timestamps[i] > cursor) fresh.push(i)
  }
  for (const i of fresh) {
    sys.ts.push(d.timestamps[i])
    sys.cpu.push(d.cpu_percent[i])
    sys.mem.push(d.memory_percent[i])
    sys.swap.push(d.swap_percent[i])
    sys.netUp.push(d.net_up_bps[i])
    sys.netDown.push(d.net_down_bps[i])
    sys.diskRead.push(d.disk_read_bps[i])
    sys.diskWrite.push(d.disk_write_bps[i])
  }
  for (const g of d.gpus) {
    if (!gpuHist[g.index]) gpuHist[g.index] = { util: [], vram: [] }
    for (const i of fresh) {
      gpuHist[g.index].util.push(g.utilization_percent[i])
      gpuHist[g.index].vram.push(g.memory_percent[i])
    }
  }
  if (d.timestamps.length) lastTs.value = d.timestamps[d.timestamps.length - 1]
  for (const col of Object.values(sys)) {
    while (col.length > MAX_POINTS) col.shift()
  }
  for (const h of Object.values(gpuHist)) {
    while (h.util.length > MAX_POINTS) h.util.shift()
    while (h.vram.length > MAX_POINTS) h.vram.shift()
  }
  snapshot.value = d.snapshot
}, { immediate: true })

const { data: status } = useQuery({
  queryKey: ['gpu-status'],
  queryFn: getGpuStatus,
  refetchInterval: 3000,
})

const loading = computed(() => historyPending.value)
const trendReady = computed(() => sys.ts.length > 1)

const isDark = useDark()
// Cache CSS tokens per theme; canvas can't resolve CSS variables, so re-read on theme switch.
const tokens = computed(() => {
  void isDark.value
  const v = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim()
  return {
    blue: v('--accent-blue'),
    cyan: v('--accent-cyan'),
    purple: v('--accent-purple'),
    green: v('--accent-green'),
    orange: v('--accent-orange'),
    textTertiary: v('--text-tertiary'),
    bgElevated: v('--bg-elevated'),
    borderDefault: v('--border-default'),
    textPrimary: v('--text-primary'),
  }
})

const last = (arr: number[]): number => (arr.length ? arr[arr.length - 1] : 0)

type ChartLike = { dispatchAction: (opt: Record<string, unknown>) => void }
const charts = new Map<string, ChartLike>()
const setChart = (key: string) => (el: unknown) => {
  if (el) charts.set(key, el as ChartLike)
  else charts.delete(key)
}
const resetZoom = (key: string) => {
  charts.get(key)?.dispatchAction({ type: 'dataZoom', start: 0, end: 100 })
}

// Aggregate per-core bars into at most CORE_SLOTS groups so machines with
// 64+ cores keep readable bars instead of sub-pixel slivers.
const CORE_SLOTS = 48
const coreBars = computed(() => {
  const cores = snapshot.value?.cpu_cores ?? []
  if (cores.length <= CORE_SLOTS) {
    return cores.map((value, i) => ({ label: `Core ${i}`, value }))
  }
  const group = Math.ceil(cores.length / CORE_SLOTS)
  const bars: { label: string; value: number }[] = []
  for (let i = 0; i < cores.length; i += group) {
    const slice = cores.slice(i, i + group)
    bars.push({
      label: `Cores ${i}–${i + slice.length - 1}`,
      value: slice.reduce((a, b) => a + b, 0) / slice.length,
    })
  }
  return bars
})

const fmtRate = (bps: number): string => {
  // Decimal (SI) units so axis ticks land on round values (1.0/2.0/5.0 MB/s).
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s']
  let v = bps
  let u = 0
  while (v >= 1000 && u < units.length - 1) {
    v /= 1000
    u++
  }
  return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ${units[u]}`
}

interface SeriesCfg {
  name: string
  color: string
  values: (number | null)[]
  fill?: boolean
}

// Round the axis ceiling up to a 1/2/5×10^n step so a single burst doesn't
// squash the whole rate chart into non-round ticks like 381.5 MB/s.
const niceMax = (values: (number | null)[]): number | undefined => {
  const peak = Math.max(0, ...values.filter((v): v is number => v != null))
  if (peak <= 0) return undefined
  const base = 10 ** Math.floor(Math.log10(peak))
  for (const m of [1, 2, 5, 10]) {
    if (m * base >= peak) return m * base
  }
  return 10 * base
}

const pctFmt = (v: number): string => `${v.toFixed(0)}%`

interface AxisCfg {
  yMax?: number
  yFmt?: (v: number) => string
}

const buildOption = (series: SeriesCfg[], axis: AxisCfg = {}) => {
  const tk = tokens.value
  const yFmt = axis.yFmt ?? pctFmt
  return {
    animation: false,
    grid: { left: 8, right: 12, top: 30, bottom: 4, containLabel: true },
    xAxis: {
      type: 'time',
      boundaryGap: false,
      axisLine: { show: false },
      axisTick: { show: false },
      // Seconds in the label: minute-only labels repeat when ticks land in the same minute.
      axisLabel: { color: tk.textTertiary, fontSize: 10, formatter: '{HH}:{mm}:{ss}', hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: axis.yMax,
      axisLabel: {
        color: tk.textTertiary,
        fontSize: 10,
        formatter: (v: number) => yFmt(v),
      },
      splitLine: { lineStyle: { color: tk.borderDefault, opacity: 0.4 } },
    },
    legend: {
      show: true,
      right: 0,
      top: 0,
      itemWidth: 10,
      itemHeight: 6,
      itemGap: 12,
      icon: 'roundRect',
      textStyle: { color: tk.textTertiary, fontSize: 10 },
    },
    dataZoom: [
      {
        type: 'inside',
        zoomOnMouseWheel: true,
        moveOnMouseWheel: true,
      },
    ],
    tooltip: {
      trigger: 'axis',
      confine: true,
      axisPointer: { type: 'line', snap: true, lineStyle: { color: tk.borderDefault } },
      backgroundColor: tk.bgElevated,
      borderColor: tk.borderDefault,
      borderWidth: 1,
      textStyle: { color: tk.textPrimary, fontSize: 11 },
      formatter: (params: unknown) => {
        const list = params as { marker: string; seriesName: string; value: [number, number | null] }[]
        if (!list.length) return ''
        const head = new Date(list[0].value[0]).toLocaleTimeString('en-GB', { hour12: false })
        const rows = list.map(
          (p) => `${p.marker} ${p.seriesName} ${p.value[1] == null ? '-' : yFmt(Number(p.value[1]))}`,
        )
        return [head, ...rows].join('<br/>')
      },
    },
    series: series.map((s) => ({
      name: s.name,
      type: 'line' as const,
      smooth: 0.3,
      symbol: 'none',
      lineStyle: { width: 1.5, color: s.color },
      // Gradient fill fades to transparent; hex alpha works because token colors are resolved hex values.
      areaStyle: s.fill
        ? {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: `${s.color}40` },
                { offset: 1, color: `${s.color}05` },
              ],
            },
          }
        : undefined,
      emphasis: { focus: 'series', blurScope: 'coordinateSystem' },
      blur: { lineStyle: { opacity: 0.25 } },
      data: sys.ts.map((t, i) => [t, s.values[i]]),
    })),
  }
}

const cpuOption = computed(() =>
  buildOption([
    { name: t('resources.cpu'), color: tokens.value.blue, values: sys.cpu, fill: true },
  ], { yMax: 100 }),
)

const memOption = computed(() =>
  buildOption([
    { name: t('resources.mem'), color: tokens.value.blue, values: sys.mem, fill: true },
    { name: t('resources.swap'), color: tokens.value.purple, values: sys.swap },
  ], { yMax: 100 }),
)

const netOption = computed(() =>
  buildOption(
    [
      { name: t('resources.netDown'), color: tokens.value.cyan, values: sys.netDown, fill: true },
      { name: t('resources.netUp'), color: tokens.value.blue, values: sys.netUp },
    ],
    { yMax: niceMax([...sys.netDown, ...sys.netUp]), yFmt: fmtRate },
  ),
)

const diskOption = computed(() =>
  buildOption(
    [
      { name: t('resources.diskRead'), color: tokens.value.green, values: sys.diskRead },
      { name: t('resources.diskWrite'), color: tokens.value.orange, values: sys.diskWrite, fill: true },
    ],
    { yMax: niceMax([...sys.diskRead, ...sys.diskWrite]), yFmt: fmtRate },
  ),
)

const gpuOption = (gpuIndex: number) => {
  const h = gpuHist[gpuIndex] ?? { util: [], vram: [] }
  return buildOption([
    { name: t('gpu.utilization'), color: tokens.value.blue, values: h.util, fill: true },
    { name: t('gpu.memory'), color: tokens.value.cyan, values: h.vram },
  ], { yMax: 100 })
}

const progressColor = (pct: number) => {
  if (pct > 90) return 'var(--accent-red)'
  if (pct > 70) return 'var(--accent-orange)'
  return 'var(--accent-green)'
}

const tempColor = (temp: number) => {
  if (temp > 85) return 'var(--accent-red)'
  if (temp > 70) return 'var(--accent-orange)'
  return 'var(--accent-green)'
}
</script>

<style scoped>
.resource-monitor {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
}

.page-title-group {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.page-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.page-sub {
  font-size: 12px;
  color: var(--text-tertiary);
}

.live-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

.live-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent-green);
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.live-text {
  font-size: 11px;
  font-weight: 600;
  color: var(--accent-green);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-family: var(--font-mono);
}

.live-badge.offline .live-dot {
  background: var(--accent-red);
  animation: none;
}

.live-badge.offline .live-text {
  color: var(--accent-red);
}

.sys-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

@media (max-width: 1100px) {
  .sys-grid {
    grid-template-columns: 1fr;
  }
}

.res-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.res-card-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
}

.res-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  letter-spacing: 0.4px;
  text-transform: uppercase;
}

.res-value {
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--text-primary);
  white-space: nowrap;
}

.res-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.res-meta {
  font-size: 11px;
  color: var(--text-tertiary);
}

.res-meta.mono {
  font-family: var(--font-mono);
  color: var(--text-secondary);
}

.core-grid {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: 44px;
}

.core-bar {
  flex: 1;
  min-width: 3px;
  background: var(--bg-overlay);
  border-radius: 1px;
  display: flex;
  align-items: flex-end;
  overflow: hidden;
  align-self: stretch;
}

.core-fill {
  width: 100%;
  border-radius: 1px;
  transition: height 0.6s ease, background 0.4s ease;
}

.chart-wrap {
  height: 200px;
}

.trend-chart {
  width: 100%;
  height: 200px;
}

.trend-placeholder {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  background: var(--bg-overlay);
  border-radius: var(--radius-sm);
}

.section-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  letter-spacing: 0.4px;
  text-transform: uppercase;
  margin-top: 4px;
}

.gpu-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}

.gpu-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.gpu-card-header {
  display: flex;
  align-items: center;
  gap: 10px;
}

.gpu-index {
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  padding: 2px 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--accent-cyan);
  white-space: nowrap;
}

.gpu-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px 20px;
}

.stat-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.stat-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-tertiary);
  letter-spacing: 0.4px;
}

.stat-value {
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--text-primary);
}

.stat-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.temp-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.progress-track {
  width: 100%;
  height: 4px;
  background: var(--bg-elevated);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.6s ease, background 0.4s ease;
}

.gpu-trend {
  height: 200px;
}

.gpu-trend .trend-chart {
  height: 200px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  gap: 8px;
}

.empty-icon {
  color: var(--text-tertiary);
}

.empty-text {
  font-size: 15px;
  font-weight: 500;
  color: var(--text-secondary);
}

.empty-sub {
  font-size: 12px;
  color: var(--text-tertiary);
}

.updated-at {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-tertiary);
  text-align: right;
}

.occupancy {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-top: 12px;
  border-top: 1px solid var(--border-muted);
}

.occupancy-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-tertiary);
  letter-spacing: 0.4px;
}

.occupancy-empty {
  font-size: 12px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
}

.occupancy-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.occ-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.occ-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--accent-blue);
}

.occ-dot.occ-running {
  background: var(--accent-blue);
  box-shadow: 0 0 5px var(--accent-blue);
}

.occ-dot.occ-stopping {
  background: var(--accent-orange);
}

.occ-dot.occ-interrupted {
  background: var(--text-tertiary);
}

.occ-name {
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

@media (prefers-reduced-motion: reduce) {
  .live-dot { animation: none; }
  .progress-fill { transition: none; }
  .core-bar { transition: none; }
}
</style>
