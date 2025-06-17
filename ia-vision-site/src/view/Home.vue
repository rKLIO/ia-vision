<template>
  <div
    class="relative flex items-center justify-center h-screen bg-linear-to-br from-blue-900 to-yellow-900 overflow-hidden"
  >
    <GlowBorder :color="['#A07CFE', '#FE8FB5', '#FFBE7B']" :border-radius="10" />
    <Eve class="z-10" />
    <div class="h-[800px] w-fit flex flex-col items-center justify-start gap-10">
      <Video class="z-10" />
      <Analyse class="z-10" />
    </div>
    <Walle class="z-10" />
    <ParticlesBg
      class="absolute inset-0"
      :quantity="300"
      :ease="100"
      :color="'#FFF'"
      :staticity="10"
      refresh
    />
  </div>
</template>

<script setup>
import { ParticlesBg } from '@/components/ui/particles-bg'
import GlowBorder from '@/components/ui/glow-border/GlowBorder.vue'
import Eve from '@/components/Eve.vue'
import Walle from '@/components/Walle.vue'
import Video from '@/components/Video.vue'
import Analyse from '@/components/Analyse.vue'
import { onMounted } from 'vue'
onMounted(() => {
  setTimeout(() => {
    const viewers = document.querySelectorAll('spline-viewer')
    viewers.forEach((viewer) => {
      const logo = viewer?.shadowRoot?.querySelector('#logo')
      logo?.remove()
    })
  }, 3000)
})
</script>
