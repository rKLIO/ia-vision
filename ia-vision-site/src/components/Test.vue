<template>
  <div
    class="relative flex h-[500px] w-full flex-col items-center justify-center overflow-hidden rounded-lg border bg-background md:shadow-xl bg-blue-500"
  >
    <ParticlesBg
      class="absolute inset-0"
      :quantity="100"
      :ease="100"
      :color="isDark ? '#FFF' : '#000'"
      :staticity="10"
      refresh
    />
  </div>
</template>

<script setup lang="ts">
import { ParticlesBg } from '@/components/ui/particles-bg'
import { computed } from 'vue'
import { useColorMode } from '@vueuse/core'

const isDark = computed(() => useColorMode().value == 'dark')
</script>
