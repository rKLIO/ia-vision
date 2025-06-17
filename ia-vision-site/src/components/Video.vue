<template>
  <div class="h-fit w-fit flex justify-start items-center">
    <div class="relative w-[calc(100vw-50em)] h-[500px] rounded-lg overflow-hidden">
      <img :src="videoUrl" alt="Flux vidéo" class="w-full h-full object-cover" />
      <!-- <GlowBorder :color="['#A07CFE', '#FE8FB5', '#FFBE7B']" :border-radius="10" /> -->
    </div>
    <!-- <pre>{{ analysisData }}</pre> -->
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import GlowBorder from '@/components/ui/glow-border/GlowBorder.vue'

const videoUrl = ref('http://localhost:5000/video_feed')
</script>
