import React, { useState, useEffect, useRef, useCallback } from 'react'
import { getConfig, setConfig } from '../lib/api'
import { isMockRequested } from '../lib/useTrustStream'
import { useToast } from '../components/Toast'
import { Card } from '../components/Card'
import { Chip } from '../components/Chip'
import { RotateCcw } from 'lucide-react'

export const SettingsPage: React.FC = () => {
  const isMock = isMockRequested()
  const { toast } = useToast()

  const [formValues, setFormValues] = useState<Record<string, number | boolean>>({
    'gate.min_sync': 0.35,
    'gate.min_margin': 0.1,
    'gate.min_directed': 0.5,
    'gate.min_quality': 0.5,
    'addressee.max_yaw_deg': 25,
    'addressee.max_pitch_deg': 20,
    'addressee.yaw_offset_deg': 0.0,
    'addressee.pitch_offset_deg': 0.0,
    'addressee.gaze_weight': 0.3,
    'addressee.use_gaze': true,
    'sync.min_lip_std': 0.004,
  })

  const debounceTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})

  const fetchConfig = useCallback(async () => {
    const res = await getConfig()
    if (res.ok) {
      const g = (res.data.gate ?? {}) as Record<string, unknown>
      const a = (res.data.addressee ?? {}) as Record<string, unknown>
      const s = (res.data.sync ?? {}) as Record<string, unknown>

      setFormValues({
        'gate.min_sync': Number(g.min_sync ?? 0.35),
        'gate.min_margin': Number(g.min_margin ?? 0.1),
        'gate.min_directed': Number(g.min_directed ?? 0.5),
        'gate.min_quality': Number(g.min_quality ?? 0.5),
        'addressee.max_yaw_deg': Number(a.max_yaw_deg ?? 25),
        'addressee.max_pitch_deg': Number(a.max_pitch_deg ?? 20),
        'addressee.yaw_offset_deg': Number(a.yaw_offset_deg ?? 0.0),
        'addressee.pitch_offset_deg': Number(a.pitch_offset_deg ?? 0.0),
        'addressee.gaze_weight': Number(a.gaze_weight ?? 0.3),
        'addressee.use_gaze': Boolean(a.use_gaze ?? true),
        'sync.min_lip_std': Number(s.min_lip_std ?? 0.004),
      })
    }
  }, [])

  useEffect(() => {
    if (!isMock) {
      fetchConfig()
    }
  }, [isMock, fetchConfig])

  const handleValueChange = (fullKey: string, newValue: number | boolean) => {
    const previousValue = formValues[fullKey]
    setFormValues((prev) => ({ ...prev, [fullKey]: newValue }))

    if (isMock) return

    if (debounceTimers.current[fullKey]) {
      clearTimeout(debounceTimers.current[fullKey])
    }

    debounceTimers.current[fullKey] = setTimeout(async () => {
      const [section, key] = fullKey.split('.')
      const res = await setConfig(section, key, newValue)
      if (res.ok) {
        toast('Applied. The pipeline re-calibrates for about 1 s.', 'accept')
      } else {
        // Revert slider
        setFormValues((prev) => ({ ...prev, [fullKey]: previousValue }))
        toast(res.message, 'reject')
      }
    }, 400)
  }

  const renderSlider = (
    fullKey: string,
    label: string,
    min: number,
    max: number,
    step: number,
    unit = '',
  ) => {
    const value = formValues[fullKey] as number
    return (
      <div className="flex flex-col gap-1.5 py-1">
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium text-text">
            {label} {unit && <span className="text-muted">({unit})</span>}
          </span>
          <span className="font-mono tabular-nums text-muted">
            {typeof value === 'number'
              ? step < 0.01
                ? value.toFixed(3)
                : step < 1
                  ? value.toFixed(2)
                  : value.toString()
              : String(value)}
            {unit}
          </span>
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => handleValueChange(fullKey, parseFloat(e.target.value))}
          className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-white/[0.1] accent-accept focus-visible:outline-2 focus-visible:outline-accept"
        />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6 max-w-4xl">
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-text">Pipeline Settings</h1>
          <p className="text-xs text-muted mt-0.5">
            Calibrate decision boundaries and addressee classification thresholds
          </p>
        </div>

        <div className="flex items-center gap-3">
          {isMock && (
            <Chip tone="info">Mock: changes are not sent</Chip>
          )}

          <button
            type="button"
            onClick={fetchConfig}
            aria-label="Reload from server"
            className="inline-flex items-center gap-2 rounded-lg border border-white/[0.08] bg-white/[0.04] px-3.5 py-1.5 text-xs font-medium text-text hover:bg-white/[0.08] focus-visible:outline-2 focus-visible:outline-accept transition-colors"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Reload from server</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Trust Gate Card */}
        <Card title="Trust gate" className="flex flex-col gap-3">
          {renderSlider('gate.min_sync', 'Min Sync Correlation', 0, 1, 0.01)}
          {renderSlider('gate.min_margin', 'Min Sync Margin', 0, 0.5, 0.01)}
          {renderSlider('gate.min_directed', 'Min Directed Score', 0, 1, 0.01)}
          {renderSlider('gate.min_quality', 'Min Face Quality', 0, 1, 0.01)}
        </Card>

        {/* Sync Card */}
        <Card title="Sync" className="flex flex-col gap-3">
          {renderSlider('sync.min_lip_std', 'Min Lip Motion (std)', 0.001, 0.03, 0.001)}
          <p className="text-[11px] text-muted mt-4 leading-relaxed">
            Mouth motion standard deviation protects against static replay attacks (phone held under chin).
          </p>
        </Card>

        {/* Addressee Card (Full width on md) */}
        <Card title="Addressee" className="md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="flex flex-col gap-3">
            {renderSlider('addressee.max_yaw_deg', 'Max Yaw Deviation', 5, 90, 1, '°')}
            {renderSlider('addressee.max_pitch_deg', 'Max Pitch Deviation', 5, 60, 1, '°')}
            {renderSlider('addressee.yaw_offset_deg', 'Yaw Offset Calibration', -30, 30, 1, '°')}
          </div>
          <div className="flex flex-col gap-3">
            {renderSlider('addressee.pitch_offset_deg', 'Pitch Offset Calibration', -30, 30, 1, '°')}
            {renderSlider('addressee.gaze_weight', 'Gaze Weight Penalty', 0.05, 1, 0.05)}

            {/* Gaze Toggle Switch */}
            <div className="flex items-center justify-between pt-2">
              <label htmlFor="use_gaze_switch" className="text-xs font-medium text-text cursor-pointer">
                Use Gaze Iris Tracking
              </label>
              <input
                id="use_gaze_switch"
                type="checkbox"
                checked={Boolean(formValues['addressee.use_gaze'])}
                onChange={(e) => handleValueChange('addressee.use_gaze', e.target.checked)}
                className="h-4 w-4 rounded border-white/[0.2] bg-white/[0.04] text-accept accent-accept focus-visible:outline-2 focus-visible:outline-accept cursor-pointer"
              />
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}
